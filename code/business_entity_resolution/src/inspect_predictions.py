import heapq
import re
import time

import pandas as pd


ROOT = "dataset"
SAMPLE_SIZE = 10
THRESHOLD = 0.65
TOP_K = 20
CHUNK_SIZE = 100_000


def read_tsv(path, usecols=None):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=usecols,
    )


def tokenize(value):
    """Convert text to lowercase letter/number tokens."""
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def jaccard(a, b):
    if not a or not b:
        return 0.0

    union = a | b
    return len(a & b) / len(union) if union else 0.0


def score_pair(qname, qaddr, cname, caddr):
    """
    Compare name and address separately.
    When either address is missing, reduce confidence in name-only matching.
    """
    name_score = jaccard(qname, cname)

    if qaddr and caddr:
        address_score = jaccard(qaddr, caddr)
        return 0.6 * name_score + 0.4 * address_score

    return 0.65 * name_score


def load_validation_sample():
    print("Loading validation sample...")

    labels = read_tsv(
        f"{ROOT}/train/validation_source1.tsv",
        usecols=["source1_entity_id", "matched_entity_ids"],
    )

    labels = labels.sample(
        n=min(SAMPLE_SIZE, len(labels)),
        random_state=42,
    ).reset_index(drop=True)

    wanted_ids = set(labels["source1_entity_id"])

    print("Finding sampled business details in train_source1.tsv...")
    details = []

    for chunk in pd.read_csv(
        f"{ROOT}/train/train_source1.tsv",
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
        found = chunk[chunk["entity_id"].isin(wanted_ids)]
        if not found.empty:
            details.append(found)

    if not details:
        raise ValueError("Could not find sampled IDs in train_source1.tsv")

    source1 = pd.concat(details, ignore_index=True)
    source1 = source1.rename(
        columns={"entity_id": "source1_entity_id"}
    )

    sample = labels.merge(
        source1,
        on="source1_entity_id",
        how="inner",
        validate="one_to_one",
    )

    if len(sample) != len(labels):
        raise ValueError(
            f"Found details for {len(sample)} of {len(labels)} sampled IDs."
        )

    print(f"Validation businesses loaded: {len(sample)}")
    return sample


def main():
    start = time.time()
    sample = load_validation_sample()

    queries = {}

    for _, row in sample.iterrows():
        query_id = row["source1_entity_id"]

        queries[query_id] = {
            "name": row["business_name"],
            "address": row["business_address"],
            "name_tokens": tokenize(row["business_name"]),
            "address_tokens": tokenize(row["business_address"]),
            "country": row["country"].strip().lower(),
            "true_ids": {
                value.strip()
                for value in row["matched_entity_ids"].split(",")
                if value.strip()
            },
        }

    # A min-heap holds only the top TOP_K candidates per query.
    # Heap entries: (score, candidate_id, name, address, country)
    top_candidates = {query_id: [] for query_id in queries}

    print("\nScanning training Source 2 and Source 3.")
    print("This may take a while because the files contain millions of rows.")

    for source_name in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"{ROOT}/train/{source_name}"
        print(f"\nChecking {source_name}...")

        rows_seen = 0

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
            for _, row in chunk.iterrows():
                rows_seen += 1

                candidate_id = row["entity_id"]
                candidate_name = row["business_name"]
                candidate_address = row["business_address"]
                candidate_country = row["country"].strip().lower()

                candidate_name_tokens = tokenize(candidate_name)
                candidate_address_tokens = tokenize(candidate_address)

                for query_id, query in queries.items():
                    # Skip only when both country values exist and disagree.
                    if (
                        query["country"]
                        and candidate_country
                        and query["country"] != candidate_country
                    ):
                        continue

                    score = score_pair(
                        query["name_tokens"],
                        query["address_tokens"],
                        candidate_name_tokens,
                        candidate_address_tokens,
                    )

                    if score < THRESHOLD:
                        continue

                    heap = top_candidates[query_id]
                    entry = (
                        score,
                        candidate_id,
                        candidate_name,
                        candidate_address,
                        candidate_country,
                    )

                    if len(heap) < TOP_K:
                        heapq.heappush(heap, entry)
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, entry)

            print(f"  Rows scanned so far: {rows_seen:,}")

    print("\n--- Sample prediction review ---")

    for query_id, query in queries.items():
        heap = top_candidates[query_id]
        ranked = sorted(heap, key=lambda item: item[0], reverse=True)

        print("\n" + "=" * 72)
        print(f"Source 1 ID: {query_id}")
        print(f"Name: {query['name']}")
        print(f"Address: {query['address']}")
        print(f"Country: {query['country']}")
        print(f"True matched IDs ({len(query['true_ids'])}): "
              f"{sorted(query['true_ids'])}")

        if not ranked:
            print("No candidates scored at or above the threshold.")
            continue

        print(f"\nTop candidates (threshold={THRESHOLD}, top {TOP_K}):")

        for score, candidate_id, name, address, country in ranked:
            label = "TRUE MATCH" if candidate_id in query["true_ids"] else "NOT IN TRUTH"

            print(f"\n  Score: {score:.4f} | {label}")
            print(f"  ID: {candidate_id}")
            print(f"  Name: {name}")
            print(f"  Address: {address}")
            print(f"  Country: {country}")

    elapsed_minutes = (time.time() - start) / 60
    print(f"\nFinished in {elapsed_minutes:.1f} minutes.")
    print("Reminder: this is a diagnostic script, not the submission matcher.")


if __name__ == "__main__":
    main()