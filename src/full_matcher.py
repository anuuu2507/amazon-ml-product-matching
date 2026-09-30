import csv
import sqlite3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "output" / "index_test.sqlite"
TRAIN_DIR = ROOT / "dataset" / "train"
VALIDATION_PATH = TRAIN_DIR / "validation_source1.tsv"
SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"

SAMPLE_SIZE = 30
MAX_RESULTS_PER_QUERY = 50


def normalize(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def load_validation():
    with VALIDATION_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        labels = list(csv.DictReader(f, delimiter="\t"))

    # Use a shuffled sample so we don't only test the first label group.
    import random
    random.Random(42).shuffle(labels)
    labels = labels[:SAMPLE_SIZE]

    needed_ids = {row["source1_entity_id"] for row in labels}
    truth = {
        row["source1_entity_id"]: {
            item.strip()
            for item in (row.get("matched_entity_ids") or "").split(",")
            if item.strip()
        }
        for row in labels
    }

    details = {}
    with SOURCE1_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["entity_id"] in needed_ids:
                details[row["entity_id"]] = row
                if len(details) == len(needed_ids):
                    break

    return labels, truth, details


def make_fts_query(name):
    tokens = normalize(name).split()
    # Search with up to 5 distinct tokens; quote each token for FTS5.
    tokens = list(dict.fromkeys(tokens))[:5]
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)


def main():
    if not DB_PATH.exists():
        print("Test index not found. Run the index benchmark first.")
        return

    labels, truth, details = load_validation()
    conn = sqlite3.connect(DB_PATH)

    total_true_ids = sum(len(ids) for ids in truth.values())
    retrieved_true_ids = 0
    total_candidates = 0
    tested = 0

    print(f"Validation businesses sampled: {len(labels)}")
    print("Searching the existing partial training index...")

    for label in labels:
        s1_id = label["source1_entity_id"]
        row = details.get(s1_id)
        if not row:
            continue

        query = make_fts_query(row.get("business_name", ""))
        if not query:
            candidates = []
        else:
            candidates = conn.execute(
                """
                SELECT entity_id, source, country, name
                FROM businesses
                WHERE name MATCH ?
                LIMIT ?
                """,
                (query, MAX_RESULTS_PER_QUERY),
            ).fetchall()

        candidate_ids = {item[0] for item in candidates}
        retrieved_true_ids += len(candidate_ids & truth[s1_id])
        total_candidates += len(candidates)
        tested += 1

        if tested <= 5:
            print(
                f"{s1_id}: {len(truth[s1_id])} true IDs; "
                f"{len(candidates)} candidates; "
                f"{len(candidate_ids & truth[s1_id])} true IDs retrieved"
            )

    conn.close()

    recall = (
        retrieved_true_ids / total_true_ids
        if total_true_ids
        else 0.0
    )
    average_candidates = total_candidates / tested if tested else 0.0

    print("\n--- Partial-index retrieval check ---")
    print(f"Businesses tested: {tested}")
    print(f"Known true IDs in sample: {total_true_ids}")
    print(f"Known true IDs retrieved: {retrieved_true_ids}")
    print(f"Recall on this partial index: {recall:.4f}")
    print(f"Average candidates per business: {average_candidates:.1f}")
    print(
        "\nNote: This is only a diagnostic. The index contains the first "
        "500,000 rows of each training comparison source, not the full corpus."
    )


if __name__ == "__main__":
    main()