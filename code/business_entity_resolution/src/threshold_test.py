import pandas as pd
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import heapq
import time

BASE = "dataset"
VALIDATION_FILE = f"{BASE}/train/validation_source1.tsv"
S1_FILE = f"{BASE}/train/train_source1.tsv"
S2_FILE = f"{BASE}/train/train_source2.tsv"
S3_FILE = f"{BASE}/train/train_source3.tsv"

TOP_K = 200
THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
FREQUENCY_LIMIT = 50000
CHUNK_SIZE = 100000
VALIDATION_LIMIT = 100  # small test run first


def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def tokens(text):
    return set(normalize(text).split())


def get_blocking_tokens(row):
    return tokens(row["business_name"]) | tokens(row["business_address"])


def jaccard(a, b):
    ta = tokens(a)
    tb = tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def address_similarity(a, b):
    a = normalize(a)
    b = normalize(b)
    if not a or not b:
        return None
    return SequenceMatcher(None, a, b).ratio()


def score_pair(s1row, candidate):
    name_score = jaccard(
        s1row["business_name"],
        candidate["business_name"]
    )
    addr_score = address_similarity(
        s1row["business_address"],
        candidate["business_address"]
    )
    if addr_score is None:
        return name_score
    return 0.4 * name_score + 0.6 * addr_score


def read_validation():
    val = pd.read_csv(
        VALIDATION_FILE,
        sep="\t",
        dtype=str
    ).fillna("")

    # Deterministic sample, shuffled so we don't just take singleton rows.
    if len(val) > VALIDATION_LIMIT:
        val = val.sample(
            n=VALIDATION_LIMIT,
            random_state=42
        ).reset_index(drop=True)

    ids = set(val["source1_entity_id"])
    labels = {}

    for _, row in val.iterrows():
        raw = row["matched_entity_ids"]
        labels[row["source1_entity_id"]] = {
            x.strip() for x in raw.split(",") if x.strip()
        }

    return val, ids, labels


def load_s1_details(ids):
    print("Loading selected S1 business details...", flush=True)
    pieces = []

    for chunk in pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        chunk = chunk.fillna("")
        selected = chunk[chunk["entity_id"].isin(ids)]
        if not selected.empty:
            pieces.append(selected)

    if not pieces:
        raise RuntimeError("Could not find validation IDs in train_source1.tsv")

    details = pd.concat(pieces, ignore_index=True)
    return {
        row["entity_id"]: row
        for _, row in details.iterrows()
    }


def count_frequencies(path, label):
    print(f"Counting token frequencies in {label}...", flush=True)
    freq = Counter()
    rows_seen = 0
    started = time.time()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        chunk = chunk.fillna("")
        for _, row in chunk.iterrows():
            freq.update(get_blocking_tokens(row))

        rows_seen += len(chunk)
        print(
            f"  {label}: scanned {rows_seen:,} rows "
            f"in {time.time() - started:.1f}s",
            flush=True
        )

    return freq


def make_token_map(s1_rows, frequencies):
    token_map = {}
    for entity_id, row in s1_rows.items():
        token_map[entity_id] = {
            t for t in get_blocking_tokens(row)
            if frequencies.get(t, 0) <= FREQUENCY_LIMIT
        }
    return token_map


def rank_source(path, label, token_map, s1_rows):
    print(f"Ranking candidates from {label}...", flush=True)

    token_to_ids = defaultdict(list)
    for s1_id, token_set in token_map.items():
        for token in token_set:
            token_to_ids[token].append(s1_id)

    # Keep a min-heap per validation business, avoiding repeated full sorting.
    heaps = {s1_id: [] for s1_id in s1_rows}
    rows_seen = 0
    started = time.time()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        chunk = chunk.fillna("")

        for _, candidate in chunk.iterrows():
            candidate_tokens = get_blocking_tokens(candidate)

            possible_ids = set()
            for token in candidate_tokens:
                possible_ids.update(token_to_ids.get(token, []))

            for s1_id in possible_ids:
                score = score_pair(s1_rows[s1_id], candidate)

                # Include a tie-breaker so heap entries are consistently ordered.
                item = (score, candidate["entity_id"])
                heap = heaps[s1_id]

                if len(heap) < TOP_K:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)

        rows_seen += len(chunk)
        print(
            f"  {label}: scanned {rows_seen:,} rows "
            f"in {time.time() - started:.1f}s",
            flush=True
        )

    return heaps


def evaluate(threshold, heaps_s2, heaps_s3, labels):
    tp = fp = fn = 0
    empty_correct = 0
    predicted_total = 0

    for s1_id, actual in labels.items():
        predicted = set()

        for score, candidate_id in heaps_s2[s1_id]:
            if score >= threshold:
                predicted.add(candidate_id)

        for score, candidate_id in heaps_s3[s1_id]:
            if score >= threshold:
                predicted.add(candidate_id)

        predicted_total += len(predicted)
        tp += len(predicted & actual)
        fp += len(predicted - actual)
        fn += len(actual - predicted)

        if not actual and not predicted:
            empty_correct += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f05 = (
        1.25 * precision * recall / (0.25 * precision + recall)
        if precision + recall else 0.0
    )

    return precision, recall, f05, empty_correct, predicted_total


def main():
    validation, ids, labels = read_validation()
    print(f"Validation businesses sampled: {len(ids)}", flush=True)

    s1_rows = load_s1_details(ids)
    ids = set(s1_rows)

    print("Counting source token frequencies...", flush=True)
    freq_s2 = count_frequencies(S2_FILE, "S2")
    freq_s3 = count_frequencies(S3_FILE, "S3")

    tokens_s2 = make_token_map(s1_rows, freq_s2)
    tokens_s3 = make_token_map(s1_rows, freq_s3)

    heaps_s2 = rank_source(S2_FILE, "S2", tokens_s2, s1_rows)
    heaps_s3 = rank_source(S3_FILE, "S3", tokens_s3, s1_rows)

    print("\n=== Threshold evaluation ===")
    print(
        "Threshold | Precision | Recall | F0.5 | "
        "Correct empty | Predicted IDs"
    )

    for threshold in THRESHOLDS:
        p, r, f, empty, count = evaluate(
            threshold,
            heaps_s2,
            heaps_s3,
            labels
        )
        print(
            f"{threshold:9.2f} | {p:9.4f} | {r:6.4f} | "
            f"{f:4.4f} | {empty:13d} | {count}"
        )

    print(
        "\nDiagnostic only: top-200 candidates per source, "
        "100 sampled validation businesses."
    )


if __name__ == "__main__":
    main()