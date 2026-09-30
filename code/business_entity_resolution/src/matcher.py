import argparse
import heapq
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DATA = Path("dataset")
OUTPUT = Path("output")

CHUNK_SIZE = 25_000

# Small limits for the smoke test. These are NOT final-run settings.
SMOKE_SOURCE1_LIMIT = 100
SMOKE_CANDIDATE_ROWS_PER_SOURCE = 100_000

MAX_TOKEN_FREQUENCY = 50_000
TOP_K_PER_SOURCE = 200
MIN_MATCH_SCORE = 0.45


def normalize_text(value):
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def tokens(value):
    return {
        token
        for token in normalize_text(value).split()
        if len(token) >= 3
    }


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_pair(query, candidate_name, candidate_address):
    name_score = jaccard(query["name"], candidate_name)

    if query["address"] and candidate_address:
        address_score = jaccard(query["address"], candidate_address)
        return 0.6 * name_score + 0.4 * address_score

    # Name-only matches are less certain when an address is missing.
    return 0.65 * name_score


def read_source1(path, limit):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=limit,
    )
    return df


def read_candidate_chunks(path):
    """Read only the first smoke-test portion of a candidate file."""
    rows_read = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        chunksize=CHUNK_SIZE,
    ):
        remaining = SMOKE_CANDIDATE_ROWS_PER_SOURCE - rows_read
        if remaining <= 0:
            break

        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining]

        rows_read += len(chunk)
        yield chunk

        if rows_read >= SMOKE_CANDIDATE_ROWS_PER_SOURCE:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small test on limited rows; this is the default.",
    )
    args = parser.parse_args()

    # Until the full matcher is optimized and validated, run only smoke test.
    if not args.smoke:
        print("This script currently supports the limited smoke test only.")
        print("Run it with: py src\\matcher.py --smoke")
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("SMOKE TEST: limited rows only; not submission output.")
    print("Loading a small Source 1 sample...")

    source1 = read_source1(
        DATA / "test" / "test_source1.tsv",
        SMOKE_SOURCE1_LIMIT,
    )
    print(f"Source 1 sample rows: {len(source1):,}")

    query_data = {}
    all_query_tokens = set()

    for row in source1.itertuples(index=False):
        name_tokens = tokens(row.business_name)
        address_tokens = tokens(row.business_address)

        all_query_tokens.update(name_tokens)
        all_query_tokens.update(address_tokens)

        query_data[row.entity_id] = {
            "name": name_tokens,
            "address": address_tokens,
            "country": normalize_text(row.country),
        }

    token_frequency = Counter()

    source_files = [
        ("S2", DATA / "test" / "test_source2.tsv"),
        ("S3", DATA / "test" / "test_source3.tsv"),
    ]

    print("\nPass 1/2: counting sample token frequencies...")
    for source_name, path in source_files:
        print(f"  Reading limited sample from {source_name}...")
        for chunk in read_candidate_chunks(path):
            for row in chunk.itertuples(index=False):
                row_tokens = (
                    tokens(row.business_name)
                    | tokens(row.business_address)
                )
                token_frequency.update(row_tokens & all_query_tokens)

    useful_tokens = {
        token
        for token, count in token_frequency.items()
        if count <= MAX_TOKEN_FREQUENCY
    }

    for query in query_data.values():
        query["name"] &= useful_tokens
        query["address"] &= useful_tokens
        query["all"] = query["name"] | query["address"]

    token_index = defaultdict(set)
    for source1_id, query in query_data.items():
        for token in query["all"]:
            token_index[token].add(source1_id)

    heaps = {
        source1_id: {"S2": [], "S3": []}
        for source1_id in query_data
    }

    print("\nPass 2/2: ranking candidates from limited samples...")
    for source_name, path in source_files:
        print(f"  Ranking limited sample from {source_name}...")
        for chunk in read_candidate_chunks(path):
            for row in chunk.itertuples(index=False):
                candidate_name = tokens(row.business_name) & useful_tokens
                candidate_address = (
                    tokens(row.business_address) & useful_tokens
                )
                candidate_all = candidate_name | candidate_address

                if not candidate_all:
                    continue

                possible_queries = set()
                for token in candidate_all:
                    possible_queries.update(token_index.get(token, ()))

                if not possible_queries:
                    continue

                candidate_country = normalize_text(row.country)

                for source1_id in possible_queries:
                    query = query_data[source1_id]

                    if (
                        query["country"]
                        and candidate_country
                        and query["country"] != candidate_country
                    ):
                        continue

                    score = score_pair(
                        query,
                        candidate_name,
                        candidate_address,
                    )

                    if score < MIN_MATCH_SCORE:
                        continue

                    heap = heaps[source1_id][source_name]
                    item = (score, row.entity_id)

                    if len(heap) < TOP_K_PER_SOURCE:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)

    result_rows = []
    candidate_rows = []

    for source1_id in source1["entity_id"]:
        selected = set()

        for source_name in ["S2", "S3"]:
            selected.update(
                candidate_id
                for score, candidate_id in heaps[source1_id][source_name]
            )

        sorted_ids = sorted(selected)

        result_rows.append({
            "source1_entity_id": source1_id,
            "matched_entity_ids": ",".join(sorted_ids),
        })

        # In this smoke test, these are the final top-ranked candidates.
        candidate_rows.append({
            "source1_entity_id": source1_id,
            "candidate_entity_ids": ",".join(sorted_ids),
        })

    matching_path = OUTPUT / "matching_results.tsv"
    candidates_path = OUTPUT / "candidate_pairs.tsv"

    pd.DataFrame(
        result_rows,
        columns=["source1_entity_id", "matched_entity_ids"],
    ).to_csv(matching_path, sep="\t", index=False)

    pd.DataFrame(
        candidate_rows,
        columns=["source1_entity_id", "candidate_entity_ids"],
    ).to_csv(candidates_path, sep="\t", index=False)

    print("\nSmoke test complete.")
    print(f"Rows written: {len(result_rows):,}")
    print(f"Matching results: {matching_path}")
    print(f"Candidate pairs: {candidates_path}")
    print("Reminder: these limited-sample files are not for submission.")


if __name__ == "__main__":
    main()