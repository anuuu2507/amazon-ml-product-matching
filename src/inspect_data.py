import pandas as pd

# 1. Load Source 1 IDs from the training-label and validation files
train_gt = pd.read_csv(
    "dataset/train/model_train_ground_truth.tsv",
    sep="\t",
    usecols=["source1_entity_id"],
    dtype=str
)

valid_gt = pd.read_csv(
    "dataset/train/validation_source1.tsv",
    sep="\t",
    usecols=["source1_entity_id"],
    dtype=str
)

# 2. Check for overlap between training and validation IDs
train_ids = set(train_gt["source1_entity_id"])
valid_ids = set(valid_gt["source1_entity_id"])

print("=== Training / Validation Check ===")
print("Training IDs:", len(train_ids))
print("Validation IDs:", len(valid_ids))
print("Overlap:", len(train_ids & valid_ids))

# 3. Check duplicate entity IDs in each training source file
print("\n=== Duplicate ID Check ===")

source_files = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv"
]

for path in source_files:
    ids = pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id"],
        dtype=str
    )["entity_id"]

    print(f"{path}")
    print("Duplicate IDs:", int(ids.duplicated().sum()))