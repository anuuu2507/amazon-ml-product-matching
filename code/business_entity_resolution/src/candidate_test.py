import pandas as pd
from pathlib import Path

DATA = Path("dataset")
SAMPLE_SIZE = 100

# Random sample from the held-out validation set
validation = pd.read_csv(
    DATA / "train" / "validation_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
).sample(n=SAMPLE_SIZE, random_state=42)

# Collect all known matched IDs for these sampled Source 1 records
true_ids = set()

for value in validation["matched_entity_ids"]:
    if value.strip():
        true_ids.update(x.strip() for x in value.split(",") if x.strip())

print("Sampled validation records:", len(validation))
print("Distinct true matched IDs:", len(true_ids))

# Check which true IDs exist in the full matching-source files
source_paths = [
    DATA / "train" / "train_source2.tsv",
    DATA / "train" / "train_source3.tsv"
]

found_ids = set()

for path in source_paths:
    print(f"Checking IDs in {path.name} ...")

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id"],
        dtype=str,
        keep_default_na=False,
        chunksize=200_000
    ):
        found_ids.update(set(chunk["entity_id"]) & true_ids)

print("True matched IDs found in source files:", len(found_ids))
print("True matched IDs missing from source files:", len(true_ids - found_ids))

if true_ids:
    print(f"Source-ID coverage: {len(found_ids) / len(true_ids):.4f}")