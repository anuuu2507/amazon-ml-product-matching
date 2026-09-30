import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

DATA = Path("dataset")
CHUNK_SIZE = 100_000
SAMPLE_SIZE = 100


def normalize_names(series):
    """Normalize names and remove common legal suffixes."""
    names = (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    suffix_pattern = (
        r"(?:\s+(?:private|pvt|limited|ltd|incorporated|inc|"
        r"corporation|corp|company|co|llc|llp))+"
        r"$"
    )

    return names.str.replace(suffix_pattern, "", regex=True).str.strip()


def main():
    validation = pd.read_csv(
        DATA / "train" / "validation_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False
    ).sample(n=SAMPLE_SIZE, random_state=42)

    validation_ids = set(validation["source1_entity_id"])

    source1 = pd.read_csv(
        DATA / "train" / "train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "business_name"]
    )

    source1 = source1[source1["entity_id"].isin(validation_ids)].copy()
    source1["name_key"] = normalize_names(source1["business_name"])

    id_to_key = dict(zip(source1["entity_id"], source1["name_key"]))
    key_to_query_ids = defaultdict(set)

    for entity_id, key in id_to_key.items():
        if key:
            key_to_query_ids[key].add(entity_id)

    true_matches = {
        row.source1_entity_id: {
            item.strip()
            for item in row.matched_entity_ids.split(",")
            if item.strip()
        }
        for row in validation.itertuples(index=False)
    }

    found_by_query = {entity_id: set() for entity_id in validation_ids}
    total_candidates = 0

    for source_name, file_name in [
        ("S2", "train_source2.tsv"),
        ("S3", "train_source3.tsv")
    ]:
        path = DATA / "train" / file_name
        print(f"\nScanning {source_name} names...")

        scanned = 0

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=["entity_id", "business_name"],
            chunksize=CHUNK_SIZE
        ):
            chunk["name_key"] = normalize_names(chunk["business_name"])
            hits = chunk[chunk["name_key"].isin(key_to_query_ids)]

            for row in hits.itertuples(index=False):
                for query_id in key_to_query_ids[row.name_key]:
                    found_by_query[query_id].add(row.entity_id)

            scanned += len(chunk)
            if scanned % 500_000 == 0:
                print(f"  Scanned {scanned:,} rows")

        print(f"Finished {source_name}: {scanned:,} rows scanned")

    total_true_ids = 0
    found_true_ids = 0

    for query_id, true_ids in true_matches.items():
        total_true_ids += len(true_ids)
        found_true_ids += len(true_ids & found_by_query[query_id])

    recall = found_true_ids / total_true_ids if total_true_ids else 0.0
    total_candidates = sum(len(ids) for ids in found_by_query.values())

    print("\n=== Exact normalized-name validation ===")
    print("Validation businesses:", len(validation))
    print("True matched IDs:", total_true_ids)
    print("True IDs found by name blocking:", found_true_ids)
    print(f"Recall: {recall:.4f}")
    print("Candidate pairs found:", total_candidates)
    print(f"Average candidates per business: {total_candidates / len(validation):.1f}")


if __name__ == "__main__":
    main()