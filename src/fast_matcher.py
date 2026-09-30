from pathlib import Path
import re
import sqlite3
import unicodedata
import difflib

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "test"
OUT = ROOT / "output"
DB_PATH = OUT / "fast_index.sqlite"

CHUNK_SIZE = 50_000
MAX_LOOKUP_ROWS = 2_000
MAX_CANDIDATES = 50
ADDRESS_MATCH_THRESHOLD = 0.92

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "plc", "gmbh", "sa", "sas", "bv", "pty", "lp"
}


def normalize(value):
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_name(value):
    words = normalize(value).split()
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


def address_similarity(a, b):
    a = normalize(a)
    b = normalize(b)
    if not a or not b:
        return None
    return difflib.SequenceMatcher(None, a, b).ratio()


def build_index(conn):
    conn.execute("DROP TABLE IF EXISTS records")
    conn.execute("""
        CREATE TABLE records (
            source TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            name_key TEXT NOT NULL,
            address TEXT,
            country TEXT
        )
    """)
    conn.commit()

    for source_num in (2, 3):
        file_path = DATA / f"test_source{source_num}.tsv"
        print(f"Indexing {file_path.name} ...")

        for chunk in pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE
        ):
            rows = []
            for row in chunk.itertuples(index=False):
                record = row._asdict()
                entity_id = record["entity_id"]
                name_key = normalize_name(record["business_name"])
                if not name_key:
                    continue

                rows.append((
                    f"S{source_num}",
                    entity_id,
                    name_key,
                    record["business_address"],
                    normalize(record["country"])
                ))

            conn.executemany(
                """
                INSERT INTO records
                (source, entity_id, name_key, address, country)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows
            )
            conn.commit()

        print(f"Finished indexing Source {source_num}.")

    print("Creating search index ...")
    conn.execute("CREATE INDEX idx_name_key ON records(name_key)")
    conn.commit()
    print("Index is ready.")


def find_candidates(conn, name_key):
    if not name_key:
        return []

    return conn.execute(
        """
        SELECT source, entity_id, address, country
        FROM records
        WHERE name_key = ?
        LIMIT ?
        """,
        (name_key, MAX_LOOKUP_ROWS)
    ).fetchall()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        print("Removing old fast index ...")
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")

    build_index(conn)

    test_s1_path = DATA / "test_source1.tsv"
    match_path = OUT / "matching_results.tsv"
    candidate_path = OUT / "candidate_pairs.tsv"

    total = 0
    with match_path.open("w", encoding="utf-8", newline="") as match_file, \
         candidate_path.open("w", encoding="utf-8", newline="") as candidate_file:

        match_file.write("source1_entity_id\tmatched_entity_ids\n")
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")

        for chunk in pd.read_csv(
            test_s1_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE
        ):
            for row in chunk.itertuples(index=False):
                record = row._asdict()
                s1_id = record["entity_id"]
                name_key = normalize_name(record["business_name"])
                s1_address = record["business_address"]
                s1_country = normalize(record["country"])

                raw_candidates = find_candidates(conn, name_key)

                ranked = []
                for source, entity_id, address, country in raw_candidates:
                    # If both countries are known, reject country conflicts.
                    if s1_country and country and s1_country != country:
                        continue

                    sim = address_similarity(s1_address, address)
                    if sim is None:
                        sim = -1.0

                    ranked.append((sim, source, entity_id, address))

                ranked.sort(key=lambda item: item[0], reverse=True)
                shortlist = ranked[:MAX_CANDIDATES]
                candidate_ids = [item[2] for item in shortlist]

                # Predict only when address evidence is strong.
                predictions = [
                    item[2] for item in shortlist
                    if item[0] >= ADDRESS_MATCH_THRESHOLD
                ]

                # If address evidence is unavailable, allow only a single
                # exact-name candidate across the retrieved records.
                if not predictions and len(shortlist) == 1:
                    only = shortlist[0]
                    if only[0] < 0:
                        predictions = [only[2]]

                match_file.write(
                    f"{s1_id}\t{','.join(predictions)}\n"
                )
                candidate_file.write(
                    f"{s1_id}\t{','.join(candidate_ids)}\n"
                )

                total += 1
                if total % 50_000 == 0:
                    print(f"Processed {total:,} Source 1 rows ...")

    conn.close()
    print(f"Done. Wrote outputs for {total:,} Source 1 rows.")
    print(f"Matching file: {match_path}")
    print(f"Candidate file: {candidate_path}")


if __name__ == "__main__":
    main()