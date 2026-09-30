import pandas as pd
from pathlib import Path

path = Path("dataset/train/train_source2.tsv")

print("Reading Source 2 in chunks...")
total_rows = 0

for chunk in pd.read_csv(
    path,
    sep="\t",
    usecols=["entity_id", "business_name", "country"],
    dtype=str,
    keep_default_na=False,
    chunksize=100_000
):
    total_rows += len(chunk)

    if total_rows >= 300_000:
        break

print("Rows checked:", total_rows)
print("Chunked read completed successfully.")