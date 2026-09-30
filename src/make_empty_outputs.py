import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_S1 = ROOT / "dataset" / "test" / "test_source1.tsv"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

matching_path = OUT / "matching_results.tsv"
candidate_path = OUT / "candidate_pairs.tsv"

with TEST_S1.open("r", encoding="utf-8-sig", newline="") as source, \
     matching_path.open("w", encoding="utf-8", newline="") as matching, \
     candidate_path.open("w", encoding="utf-8", newline="") as candidates:

    reader = csv.DictReader(source, delimiter="\t")
    match_writer = csv.writer(matching, delimiter="\t", lineterminator="\n")
    candidate_writer = csv.writer(candidates, delimiter="\t", lineterminator="\n")

    match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
    candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

    count = 0
    for row in reader:
        s1_id = row["entity_id"]
        match_writer.writerow([s1_id, ""])
        candidate_writer.writerow([s1_id, ""])
        count += 1

print(f"Created full-size empty baseline for {count:,} Source 1 rows.")
print(f"Matching file: {matching_path}")
print(f"Candidate file: {candidate_path}")
