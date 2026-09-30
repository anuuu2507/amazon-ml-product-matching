from pathlib import Path
import re
import unicodedata

import pandas as pd


# Project folder is the parent of the src folder.
PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_SOURCE1 = PROJECT_DIR / "dataset" / "test" / "test_source1.tsv"


# Generic business/legal terms and common address words that are usually
# less useful for identifying a particular business.
BLOCKING_STOPWORDS = {
    # Business/legal suffixes
    "limited", "private", "llc", "inc", "ltd", "corp",
    "corporation", "company", "co", "plc", "pvt", "llp",
    "gmbh", "sarl", "sas",

    # Common English address words
    "road", "rd", "street", "st", "drive", "dr",
    "avenue", "ave", "floor", "unit", "suite", "no",
    "new", "north", "south", "east", "west", "city",

    # Common French address/filler words
    "de", "la", "du", "des", "le", "les",
    "rue", "boulevard",
}


def normalize_text(value):
    """Lowercase text, remove accents, and keep letters/numbers."""
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    return text


def tokenize(value):
    """Split normalized text into alphanumeric tokens."""
    text = normalize_text(value)
    return re.findall(r"[a-z0-9]+", text)


def get_raw_tokens(business_name, business_address):
    """Combine name and address tokens without filtering."""
    return sorted(
        set(tokenize(business_name))
        | set(tokenize(business_address))
    )


def get_blocking_tokens(business_name, business_address):
    """
    Remove common business/address words and single-character tokens.
    Keep distinctive words and numbers for candidate blocking.
    """
    raw_tokens = get_raw_tokens(business_name, business_address)

    return [
        token
        for token in raw_tokens
        if token not in BLOCKING_STOPWORDS
        and len(token) > 1
    ]


def main():
    print("Checking raw tokens and filtered blocking tokens...")

    if not TEST_SOURCE1.exists():
        print(f"ERROR: Could not find dataset file:\n{TEST_SOURCE1}")
        return

    # Read a few rows only; this is a quick token-filter check.
    df = pd.read_csv(
        TEST_SOURCE1,
        sep="\t",
        dtype=str,
        nrows=5,
        keep_default_na=False,
    )

    required_columns = {
        "entity_id",
        "business_name",
        "business_address",
    }

    missing = required_columns - set(df.columns)
    if missing:
        print(f"ERROR: Missing required columns: {sorted(missing)}")
        print(f"Available columns: {list(df.columns)}")
        return

    for _, row in df.iterrows():
        entity_id = row["entity_id"]
        business_name = row["business_name"]
        business_address = row["business_address"]

        raw_tokens = get_raw_tokens(
            business_name,
            business_address,
        )
        blocking_tokens = get_blocking_tokens(
            business_name,
            business_address,
        )

        print()
        print(f"Entity ID: {entity_id}")
        print(f"Business: {business_name}")
        print(f"Raw tokens: {raw_tokens}")
        print(f"Blocking tokens: {blocking_tokens}")

    print()
    print("Token filter check complete.")


if __name__ == "__main__":
    main()