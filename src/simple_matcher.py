from pathlib import Path
import re
import unicodedata

import pandas as pd


# Project root: ...\student_resource
ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
VALIDATION_FILE = TRAIN_DIR / "validation_source1.tsv"
SOURCE1_FILE = TRAIN_DIR / "train_source1.tsv"
GROUND_TRUTH_FILE = TRAIN_DIR / "train_ground_truth.tsv"

SOURCE_FILES = [
    TRAIN_DIR / "train_source2.tsv",
    TRAIN_DIR / "train_source3.tsv",
]

CHUNK_SIZE = 100_000


def normalize_text(value):
    """Lowercase text, remove punctuation, and standardize whitespace."""
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def make_key(name, country):
    """Create a lookup key from normalized business name and country."""
    return (normalize_text(name), normalize_text(country))


def calculate_f05(true_ids, predicted_ids):
    """Calculate F0.5 for one business, including empty-answer cases."""
    true_ids = set(true_ids)
    predicted_ids = set(predicted_ids)

    # Correctly predicting no matches for a singleton is a full score.
    if not true_ids and not predicted_ids:
        return 1.0

    if not true_ids or not predicted_ids:
        return 0.0

    correct = len(true_ids.intersection(predicted_ids))
    precision = correct / len(predicted_ids)
    recall = correct / len(true_ids)

    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0

    return 1.25 * precision * recall / denominator


def main():
    # Check that the required files exist before starting.
    required_files = [
        VALIDATION_FILE,
        SOURCE1_FILE,
        GROUND_TRUTH_FILE,
        *SOURCE_FILES,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    print("Loading validation IDs and labels...")
    validation = pd.read_csv(
        VALIDATION_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required_validation_columns = {
        "source1_entity_id",
        "matched_entity_ids",
    }
    missing = required_validation_columns - set(validation.columns)
    if missing:
        raise ValueError(
            f"Validation file is missing columns: {sorted(missing)}"
        )

    print("Loading Source 1 business details...")
    source1 = pd.read_csv(
        SOURCE1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    print("Joining validation IDs to Source 1 details...")
    validation = validation.merge(
        source1,
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left",
        validate="one_to_one",
    )

    if validation["business_name"].isna().any():
        missing_count = int(validation["business_name"].isna().sum())
        raise ValueError(
            f"{missing_count} validation IDs were not found in Source 1."
        )

    print(f"Validation businesses loaded: {len(validation):,}")

    # Prepare normalized Source 1 fields once.
    validation["normalized_name"] = validation["business_name"].map(
        normalize_text
    )
    validation["normalized_country"] = validation["country"].map(
        normalize_text
    )
    validation["normalized_address"] = validation["business_address"].map(
        normalize_text
    )

    # Map each (name, country) key to its validation row index.
    key_to_validation_rows = {}

    for row_number, row in enumerate(
        validation[
            ["normalized_name", "normalized_country"]
        ].itertuples(index=False, name=None)
    ):
        name_key, country_key = row

        if name_key and country_key:
            key = (name_key, country_key)
            key_to_validation_rows.setdefault(key, []).append(row_number)

    print(
        "Distinct validation name/country keys:",
        f"{len(key_to_validation_rows):,}",
    )

    # Keep the predictions for each validation business.
    predictions = [set() for _ in range(len(validation))]

    # Scan Source 2 and Source 3 in chunks to limit memory use.
    for source_path in SOURCE_FILES:
        print(f"\nScanning {source_path.name}...")
        rows_scanned = 0
        hits_found = 0

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ],
        ):
            rows_scanned += len(chunk)

            chunk["normalized_name"] = chunk["business_name"].map(
                normalize_text
            )
            chunk["normalized_country"] = chunk["country"].map(
                normalize_text
            )
            chunk["normalized_address"] = chunk["business_address"].map(
                normalize_text
            )

            # Only rows whose name and country key occurs in validation
            # need to be checked further.
            for row in chunk.itertuples(index=False):
                key = (row.normalized_name, row.normalized_country)

                if not key[0] or not key[1]:
                    continue

                validation_rows = key_to_validation_rows.get(key)
                if not validation_rows:
                    continue

                # This prototype requires both addresses to be non-empty
                # and exactly equal after normalization.
                candidate_address = row.normalized_address
                if not candidate_address:
                    continue

                for validation_row in validation_rows:
                    validation_address = validation.iloc[
                        validation_row
                    ]["normalized_address"]

                    if (
                        validation_address
                        and validation_address == candidate_address
                    ):
                        predictions[validation_row].add(row.entity_id)
                        hits_found += 1

            print(
                f"  Scanned {rows_scanned:,} rows; "
                f"exact name/country/address hits: {hits_found:,}"
            )

    # Evaluate the prototype against the validation labels.
    print("\nEvaluating predictions...")

    per_business_scores = []
    total_true_ids = 0
    total_retrieved_true_ids = 0
    businesses_with_predictions = 0
    businesses_with_no_true_matches = 0
    correct_empty_predictions = 0
    total_predicted_ids = 0

    for row_number, row in validation.iterrows():
        true_text = row["matched_entity_ids"]

        true_ids = {
            item.strip()
            for item in true_text.split(",")
            if item.strip()
        }
        predicted_ids = predictions[row_number]

        total_true_ids += len(true_ids)
        total_retrieved_true_ids += len(true_ids.intersection(predicted_ids))
        total_predicted_ids += len(predicted_ids)

        if predicted_ids:
            businesses_with_predictions += 1

        if not true_ids:
            businesses_with_no_true_matches += 1
            if not predicted_ids:
                correct_empty_predictions += 1

        per_business_scores.append(
            calculate_f05(true_ids, predicted_ids)
        )

    mean_f05 = (
        sum(per_business_scores) / len(per_business_scores)
        if per_business_scores
        else 0.0
    )

    candidate_recall = (
        total_retrieved_true_ids / total_true_ids
        if total_true_ids
        else 0.0
    )

    average_predictions = (
        total_predicted_ids / len(validation)
        if len(validation)
        else 0.0
    )

    print("\n===== VALIDATION SUMMARY =====")
    print(f"Validation businesses: {len(validation):,}")
    print(
        "Businesses with at least one prediction:",
        f"{businesses_with_predictions:,}",
    )
    print(f"True matched IDs: {total_true_ids:,}")
    print(f"True IDs retrieved: {total_retrieved_true_ids:,}")
    print(f"Candidate recall: {candidate_recall:.4f}")
    print(f"Total predicted IDs: {total_predicted_ids:,}")
    print(f"Average predictions per business: {average_predictions:.2f}")
    print(
        "Validation businesses with no true matches:",
        f"{businesses_with_no_true_matches:,}",
    )
    print(
        "Correct empty predictions:",
        f"{correct_empty_predictions:,}",
    )
    print(f"Mean per-business F0.5: {mean_f05:.4f}")
    print(
        "\nNote: this is a strict exact-match prototype on the validation "
        "sample, not a final matcher or official leaderboard score."
    )


if __name__ == "__main__":
    main()