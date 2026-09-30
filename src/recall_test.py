from pathlib import Path
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "dataset" / "train"

VALIDATION_FILE = TRAIN_DIR / "validation_source1.tsv"
SOURCE1_FILE = TRAIN_DIR / "train_source1.tsv"
SOURCE2_FILE = TRAIN_DIR / "train_source2.tsv"
SOURCE3_FILE = TRAIN_DIR / "train_source3.tsv"

SAMPLE_SIZE = 100
SCAN_LIMIT_PER_SOURCE = 100_000
CHUNK_SIZE = 25_000
MAX_TOKEN_FREQUENCY = 50_000

STOPWORDS = {
    "limited", "private", "llc", "inc", "ltd", "corp",
    "corporation", "company", "co", "plc", "pvt", "llp",
    "gmbh", "sarl", "sas",
    "road", "rd", "street", "st", "drive", "dr",
    "avenue", "ave", "floor", "unit", "suite", "no",
    "new", "north", "south", "east", "west", "city",
    "de", "la", "du", "des", "le", "les", "rue", "boulevard",
}


def normalize(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = unicodedata.normalize("NFKD", value)

    return "".join(
        char for char in value
        if not unicodedata.combining(char)
    )


def get_tokens(name, address):
    text = normalize(name) + " " + normalize(address)
    found = set(re.findall(r"[a-z0-9]+", text))

    return {
        token for token in found
        if len(token) > 1 and token not in STOPWORDS
    }


def read_tsv(path, **kwargs):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        **kwargs,
    )


def main():
    for path in [
        VALIDATION_FILE,
        SOURCE1_FILE,
        SOURCE2_FILE,
        SOURCE3_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    print("Loading validation sample...")
    validation_all = read_tsv(VALIDATION_FILE)

    needed_validation_cols = {
        "source1_entity_id",
        "matched_entity_ids",
    }
    missing = needed_validation_cols - set(validation_all.columns)

    if missing:
        raise ValueError(
            "Unexpected validation file columns. "
            f"Missing: {sorted(missing)}. "
            f"Found: {list(validation_all.columns)}"
        )

    # Sample validation rows, keeping the matching labels with each ID.
    validation = validation_all.sample(
        n=min(SAMPLE_SIZE, len(validation_all)),
        random_state=42,
    ).reset_index(drop=True)

    # Rename the validation ID to match Source 1's entity_id column.
    validation = validation.rename(
        columns={"source1_entity_id": "entity_id"}
    )

    print("Joining validation IDs to Source 1 business details...")
    source1 = read_tsv(
        SOURCE1_FILE,
        usecols=["entity_id", "business_name", "business_address"],
    )

    validation = validation.merge(
        source1,
        on="entity_id",
        how="left",
        validate="one_to_one",
    )

    if validation["business_name"].isna().any():
        missing_count = validation["business_name"].isna().sum()
        raise ValueError(
            f"{missing_count} validation IDs were not found in train_source1.tsv."
        )

    validation["tokens"] = validation.apply(
        lambda row: get_tokens(
            row["business_name"],
            row["business_address"],
        ),
        axis=1,
    )

    print(f"Validation businesses in sample: {len(validation)}")

    # Count token frequency in the first 100,000 rows of each comparison source.
    token_counts = Counter()
    source_files = [SOURCE2_FILE, SOURCE3_FILE]

    print("Counting token frequencies in limited source scans...")
    for source_file in source_files:
        rows_seen = 0

        for chunk in pd.read_csv(
            source_file,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=["entity_id", "business_name", "business_address"],
            chunksize=CHUNK_SIZE,
        ):
            for name, address in zip(
                chunk["business_name"],
                chunk["business_address"],
            ):
                token_counts.update(get_tokens(name, address))

            rows_seen += len(chunk)
            if rows_seen >= SCAN_LIMIT_PER_SOURCE:
                break

        print(f"  Scanned {rows_seen:,} rows from {source_file.name}")

    # Build token -> validation business lookup.
    token_to_queries = defaultdict(list)

    for _, row in validation.iterrows():
        query_id = row["entity_id"]
        useful_tokens = {
            token for token in row["tokens"]
            if token_counts[token] <= MAX_TOKEN_FREQUENCY
        }

        for token in useful_tokens:
            token_to_queries[token].append(query_id)

    candidate_sets = {
        query_id: set()
        for query_id in validation["entity_id"]
    }

    print("Collecting candidates from limited source scans...")
    for source_file in source_files:
        rows_seen = 0

        for chunk in pd.read_csv(
            source_file,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=["entity_id", "business_name", "business_address"],
            chunksize=CHUNK_SIZE,
        ):
            for candidate_id, name, address in zip(
                chunk["entity_id"],
                chunk["business_name"],
                chunk["business_address"],
            ):
                row_tokens = get_tokens(name, address)

                for token in row_tokens:
                    for query_id in token_to_queries.get(token, []):
                        candidate_sets[query_id].add(candidate_id)

            rows_seen += len(chunk)
            if rows_seen >= SCAN_LIMIT_PER_SOURCE:
                break

        print(f"  Scanned {rows_seen:,} rows from {source_file.name}")

    # Compare retrieved candidates against labels stored in validation file.
    total_true_ids = 0
    total_retrieved_ids = 0
    businesses_with_matches = 0

    for _, row in validation.iterrows():
        query_id = row["entity_id"]
        true_ids = {
            item.strip()
            for item in row["matched_entity_ids"].split(",")
            if item.strip()
        }

        if true_ids:
            businesses_with_matches += 1
            total_true_ids += len(true_ids)
            total_retrieved_ids += len(
                true_ids & candidate_sets[query_id]
            )

    recall = (
        total_retrieved_ids / total_true_ids
        if total_true_ids
        else 0.0
    )

    average_candidates = (
        sum(len(values) for values in candidate_sets.values())
        / len(candidate_sets)
        if candidate_sets
        else 0.0
    )

    print("\n--- Sample candidate retrieval report ---")
    print(f"Validation businesses sampled: {len(validation)}")
    print(f"Businesses with known matches: {businesses_with_matches}")
    print(f"Known matching IDs: {total_true_ids}")
    print(f"Known matching IDs retrieved: {total_retrieved_ids}")
    print(f"Candidate recall: {recall:.4f}")
    print(f"Average candidates per business: {average_candidates:.1f}")
    print(
        "\nNote: only the first 100,000 rows of each comparison source "
        "were scanned. This is a diagnostic check, not full-corpus recall."
    )


if __name__ == "__main__":
    main()