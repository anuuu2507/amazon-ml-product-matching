import pandas as pd
import re
import heapq
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher

DATA = Path("dataset")
SAMPLE_SIZE = 100
CHUNK_SIZE = 100_000
MAX_TOKEN_FREQUENCY = 50_000
TOP_K_PER_SOURCE = 200
SHOW_EXAMPLES = 10


def normalize_text(value):
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def tokens(value):
    return {
        token for token in normalize_text(value).split()
        if len(token) >= 3
    }


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def address_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def main():
    validation = pd.read_csv(
        DATA / "train" / "validation_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False
    ).sample(n=SAMPLE_SIZE, random_state=42)

    source1 = pd.read_csv(
        DATA / "train" / "train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "business_name", "business_address"]
    )

    source1 = source1[
        source1["entity_id"].isin(validation["source1_entity_id"])
    ]

    query_data = {}
    for row in source1.itertuples(index=False):
        query_data[row.entity_id] = {
            "name": tokens(row.business_name),
            "address": row.business_address,
            "address_tokens": tokens(row.business_address),
        }

    truth = {}
    for row in validation.itertuples(index=False):
        truth[row.source1_entity_id] = {
            value.strip()
            for value in row.matched_entity_ids.split(",")
            if value.strip()
        }

    all_query_tokens = set()
    for info in query_data.values():
        all_query_tokens.update(info["name"])
        all_query_tokens.update(info["address_tokens"])

    token_frequency = Counter()

    for file_name in ["train_source2.tsv", "train_source3.tsv"]:
        print(f"Counting token frequencies in {file_name}...")
        for chunk in pd.read_csv(
            DATA / "train" / file_name,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=["business_name", "business_address"],
            chunksize=CHUNK_SIZE
        ):
            for row in chunk.itertuples(index=False):
                row_tokens = (
                    tokens(row.business_name)
                    | tokens(row.business_address)
                )
                token_frequency.update(row_tokens & all_query_tokens)

    useful_tokens = {
        token for token, frequency in token_frequency.items()
        if frequency <= MAX_TOKEN_FREQUENCY
    }

    name_index = defaultdict(set)
    address_index = defaultdict(set)

    for source1_id, info in query_data.items():
        for token in info["name"] & useful_tokens:
            name_index[token].add(source1_id)
        for token in info["address_tokens"] & useful_tokens:
            address_index[token].add(source1_id)

    heaps = {
        source1_id: {"S2": [], "S3": []}
        for source1_id in query_data
    }

    for source_name, file_name in [
        ("S2", "train_source2.tsv"),
        ("S3", "train_source3.tsv")
    ]:
        print(f"Ranking candidates from {file_name}...")

        for chunk in pd.read_csv(
            DATA / "train" / file_name,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=["entity_id", "business_name", "business_address"],
            chunksize=CHUNK_SIZE
        ):
            for row in chunk.itertuples(index=False):
                candidate_name_tokens = tokens(row.business_name)
                candidate_address_tokens = tokens(row.business_address)
                candidate_tokens = (
                    candidate_name_tokens | candidate_address_tokens
                ) & useful_tokens

                possible_queries = set()
                for token in candidate_tokens:
                    possible_queries.update(name_index.get(token, ()))
                    possible_queries.update(address_index.get(token, ()))

                for source1_id in possible_queries:
                    query = query_data[source1_id]

                    name_score = jaccard(
                        query["name"], candidate_name_tokens
                    )
                    address_score = address_similarity(
                        query["address"], row.business_address
                    )
                    score = 0.4 * name_score + 0.6 * address_score

                    heap = heaps[source1_id][source_name]
                    item = (score, row.entity_id)

                    if len(heap) < TOP_K_PER_SOURCE:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)

        print(f"Finished {source_name}.")

    missed = []

    for source1_id, true_ids in truth.items():
        shortlisted = set()
        for source_name in ["S2", "S3"]:
            shortlisted.update(
                entity_id
                for score, entity_id in heaps[source1_id][source_name]
            )

        for true_id in true_ids - shortlisted:
            missed.append((source1_id, true_id))

    print(f"\nMissed true matches in this sample: {len(missed)}")
    print(f"Showing up to {SHOW_EXAMPLES} examples:\n")

    if not missed:
        print("No missed true matches in this sample.")
        return

    source1_lookup = source1.set_index("entity_id").to_dict("index")

    # Read only the source files needed to look up missed IDs.
    missed_by_source = {"S2": set(), "S3": set()}
    for _, true_id in missed:
        if true_id.startswith("S2-"):
            missed_by_source["S2"].add(true_id)
        elif true_id.startswith("S3-"):
            missed_by_source["S3"].add(true_id)

    match_rows = {}

    for source_name, file_name in [
        ("S2", "train_source2.tsv"),
        ("S3", "train_source3.tsv")
    ]:
        wanted = missed_by_source[source_name]
        if not wanted:
            continue

        for chunk in pd.read_csv(
            DATA / "train" / file_name,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=["entity_id", "business_name", "business_address"],
            chunksize=CHUNK_SIZE
        ):
            found = chunk[chunk["entity_id"].isin(wanted)]
            for row in found.itertuples(index=False):
                match_rows[row.entity_id] = {
                    "source": source_name,
                    "name": row.business_name,
                    "address": row.business_address
                }

    for source1_id, true_id in missed[:SHOW_EXAMPLES]:
        s1 = source1_lookup[source1_id]
        match = match_rows.get(true_id, {})

        print("-" * 70)
        print("Source 1 ID:", source1_id)
        print("Source 1 name:", s1["business_name"])
        print("Source 1 address:", s1["business_address"])
        print("Missed true ID:", true_id)
        print("Matched source:", match.get("source", "Unknown"))
        print("Matched name:", match.get("name", "Not found"))
        print("Matched address:", match.get("address", "Not found"))


if __name__ == "__main__":
    main()