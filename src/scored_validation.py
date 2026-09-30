from pathlib import Path
import re
import unicodedata
import heapq

import pandas as pd


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "dataset" / "train"

SOURCE1_FILE = TRAIN_DIR / "train_source1.tsv"
SOURCE2_FILE = TRAIN_DIR / "train_source2.tsv"
SOURCE3_FILE = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH_FILE = TRAIN_DIR / "train_ground_truth.tsv"
VALIDATION_FILE = TRAIN_DIR / "validation_source1.tsv"

SAMPLE_SIZE = 100
CHUNK_SIZE = 25_000
TOP_PER_SOURCE = 100

TOP_K_VALUES = [1, 3, 5, 10, 20]
THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

STOPWORDS = {
    "the", "and", "for", "with", "from",
    "inc", "incorporated", "llc", "ltd", "limited",
    "corp", "corporation", "company", "co",
    "plc", "gmbh", "sa", "sarl", "pty",
    "street", "st", "road", "rd", "avenue", "ave",
    "boulevard", "blvd", "lane", "ln", "drive", "dr",
    "building", "bldg", "floor", "suite", "unit",
    "address", "unknown", "none", "null",
}


# ---------------------------------------------------------
# Text processing
# ---------------------------------------------------------

def normalize_text(value):
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_tokens(value):
    normalized = normalize_text(value)

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= 2 and token not in STOPWORDS
    }


def jaccard_similarity(tokens_a, tokens_b):
    if not tokens_a or not tokens_b:
        return 0.0

    union = tokens_a | tokens_b
    if not union:
        return 0.0

    return len(tokens_a & tokens_b) / len(union)


def score_candidate(query, candidate):
    query_country = normalize_text(query["country"])
    candidate_country = normalize_text(candidate["country"])

    if (
        query_country
        and candidate_country
        and query_country != candidate_country
    ):
        return 0.0

    name_score = jaccard_similarity(
        query["name_tokens"],
        candidate["name_tokens"],
    )

    query_address_tokens = query["address_tokens"]
    candidate_address_tokens = candidate["address_tokens"]

    if query_address_tokens and candidate_address_tokens:
        address_score = jaccard_similarity(
            query_address_tokens,
            candidate_address_tokens,
        )
        return 0.60 * name_score + 0.40 * address_score

    return name_score


# ---------------------------------------------------------
# Load validation sample and labels
# ---------------------------------------------------------

def load_validation():
    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}"
        )

    validation_ids = pd.read_csv(
        VALIDATION_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    if "source1_entity_id" in validation_ids.columns:
        id_column = "source1_entity_id"
    elif "entity_id" in validation_ids.columns:
        id_column = "entity_id"
    else:
        raise ValueError(
            "Validation file needs source1_entity_id or entity_id."
        )

    validation_ids = validation_ids[[id_column]].copy()
    validation_ids = validation_ids.rename(
        columns={id_column: "source1_entity_id"}
    )

    validation_ids = validation_ids.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    validation_ids = validation_ids.head(SAMPLE_SIZE)

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

    source1 = source1.rename(
        columns={"entity_id": "source1_entity_id"}
    )

    validation = validation_ids.merge(
        source1,
        on="source1_entity_id",
        how="left",
        validate="one_to_one",
    )

    if validation["business_name"].isna().any():
        missing = validation[
            validation["business_name"].isna()
        ]["source1_entity_id"].head(5).tolist()

        raise ValueError(
            f"Validation IDs missing from Source 1. Examples: {missing}"
        )

    validation["name_tokens"] = validation["business_name"].map(
        get_tokens
    )
    validation["address_tokens"] = validation["business_address"].map(
        get_tokens
    )

    ground_truth_df = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    ground_truth = dict(
        zip(
            ground_truth_df["source1_entity_id"],
            ground_truth_df["matched_entity_ids"],
        )
    )

    truth_sets = []

    for source1_id in validation["source1_entity_id"]:
        raw_ids = ground_truth.get(source1_id, "").strip()

        if not raw_ids:
            truth_sets.append(set())
        else:
            truth_sets.append({
                item.strip()
                for item in raw_ids.split(",")
                if item.strip()
            })

    validation["truth_set"] = truth_sets
    return validation


# ---------------------------------------------------------
# Candidate lookup by token
# ---------------------------------------------------------

def build_token_lookup(validation):
    token_to_queries = {}

    for row_number, row in validation.iterrows():
        combined_tokens = (
            row["name_tokens"] | row["address_tokens"]
        )

        for token in combined_tokens:
            token_to_queries.setdefault(token, []).append(row_number)

    return token_to_queries


# ---------------------------------------------------------
# Scan a source file and keep top candidates
# ---------------------------------------------------------

def scan_source(source_path, validation, token_to_queries):
    heaps = [[] for _ in range(len(validation))]
    seen_ids = [set() for _ in range(len(validation))]

    rows_seen = 0
    candidate_hits = 0
    unique_counter = 0

    usecols = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    print(f"\nScanning {source_path.name}...")

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
    ):
        rows_seen += len(chunk)

        for row in chunk.itertuples(index=False, name=None):
            candidate_id, name, address, country = row

            name_tokens = get_tokens(name)
            address_tokens = get_tokens(address)
            combined_tokens = name_tokens | address_tokens

            if not combined_tokens:
                continue

            possible_queries = set()

            for token in combined_tokens:
                query_rows = token_to_queries.get(token)
                if query_rows:
                    possible_queries.update(query_rows)

            if not possible_queries:
                continue

            candidate = {
                "country": country,
                "name_tokens": name_tokens,
                "address_tokens": address_tokens,
            }

            for query_index in possible_queries:
                if candidate_id in seen_ids[query_index]:
                    continue

                query = validation.iloc[query_index]
                score = score_candidate(query, candidate)

                if score <= 0:
                    continue

                candidate_hits += 1
                seen_ids[query_index].add(candidate_id)

                heap = heaps[query_index]
                unique_counter += 1
                item = (score, unique_counter, candidate_id)

                if len(heap) < TOP_PER_SOURCE:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)

        if rows_seen % 250_000 < CHUNK_SIZE:
            print(
                f"  Rows scanned: {rows_seen:,} | "
                f"candidate comparisons: {candidate_hits:,}"
            )

    print(
        f"Finished {source_path.name}: "
        f"{rows_seen:,} rows scanned; "
        f"{candidate_hits:,} positive-scoring comparisons."
    )

    ranked = []

    for heap in heaps:
        best = sorted(
            heap,
            key=lambda item: item[0],
            reverse=True,
        )

        ranked.append([
            (score, candidate_id)
            for score, _, candidate_id in best
        ])

    return ranked


# ---------------------------------------------------------
# Per-business metric
# ---------------------------------------------------------

def f05_for_one_business(predicted, truth):
    if not truth:
        return 1.0 if not predicted else 0.0

    if not predicted:
        return 0.0

    true_positives = len(predicted & truth)
    precision = true_positives / len(predicted)
    recall = true_positives / len(truth)

    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0

    return 1.25 * precision * recall / denominator


# ---------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------

def evaluate_thresholds(validation, combined_rankings):
    total_true_ids = sum(
        len(truth)
        for truth in validation["truth_set"]
    )

    print("\n===== SCORE THRESHOLD TEST =====")
    print(f"Validation businesses: {len(validation):,}")
    print(f"Total true matched IDs: {total_true_ids:,}")

    for threshold in THRESHOLDS:
        business_scores = []
        correct_ids = 0
        predicted_ids = 0

        for index, row in validation.iterrows():
            truth = row["truth_set"]

            predicted = {
                candidate_id
                for score, candidate_id in combined_rankings[index]
                if score >= threshold
            }

            correct_ids += len(predicted & truth)
            predicted_ids += len(predicted)

            business_scores.append(
                f05_for_one_business(predicted, truth)
            )

        mean_f05 = (
            sum(business_scores) / len(business_scores)
            if business_scores else 0.0
        )

        recall = (
            correct_ids / total_true_ids
            if total_true_ids else 0.0
        )

        precision = (
            correct_ids / predicted_ids
            if predicted_ids else 0.0
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"mean per-business F0.5: {mean_f05:.4f} | "
            f"precision: {precision:.4f} | "
            f"recall: {recall:.4f} | "
            f"predicted IDs: {predicted_ids:,}"
        )


# ---------------------------------------------------------
# Top-K evaluation and examples
# ---------------------------------------------------------

def evaluate_shortlists(validation, combined_rankings):
    all_true_ids = set()

    for truth in validation["truth_set"]:
        all_true_ids.update(truth)

    top20_retrieved = set()

    for ranked in combined_rankings:
        for _, candidate_id in ranked[:20]:
            top20_retrieved.add(candidate_id)

    top20_recall = (
        len(all_true_ids & top20_retrieved) / len(all_true_ids)
        if all_true_ids else 0.0
    )

    print("\n===== RANKED SHORTLIST REPORT =====")
    print(f"Validation businesses: {len(validation):,}")
    print(
        "Businesses with true matches: "
        f"{sum(bool(x) for x in validation['truth_set']):,}"
    )
    print(f"Distinct true matched IDs: {len(all_true_ids):,}")
    print(
        "Distinct true IDs present in any top-20 shortlist: "
        f"{len(all_true_ids & top20_retrieved):,}"
    )
    print(f"Distinct-ID top-20 recall: {top20_recall:.4f}")

    print("\nTop-K metrics (mean per-business F0.5):")

    total_truth_count = sum(
        len(truth)
        for truth in validation["truth_set"]
    )

    for k in TOP_K_VALUES:
        scores = []
        retrieved_truth_count = 0

        for index, row in validation.iterrows():
            truth = row["truth_set"]
            ranked = combined_rankings[index]

            predicted = {
                candidate_id
                for _, candidate_id in ranked[:k]
            }

            retrieved_truth_count += len(predicted & truth)
            scores.append(
                f05_for_one_business(predicted, truth)
            )

        mean_f05 = (
            sum(scores) / len(scores)
            if scores else 0.0
        )

        recall = (
            retrieved_truth_count / total_truth_count
            if total_truth_count else 0.0
        )

        print(
            f"  Top {k:>2}: "
            f"mean per-business F0.5 = {mean_f05:.4f} | "
            f"matched-ID recall = {recall:.4f}"
        )

    print("\nSample ranked results:")

    for index in range(min(5, len(validation))):
        row = validation.iloc[index]
        top_items = combined_rankings[index][:5]

        print(f"\nSource1 ID: {row['source1_entity_id']}")
        print(f"Name: {row['business_name']}")
        print(f"True IDs: {sorted(row['truth_set'])}")
        print(
            "Top candidates: "
            + str([
                (candidate_id, round(score, 3))
                for score, candidate_id in top_items
            ])
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("Loading validation sample...")
    validation = load_validation()

    print(
        f"Loaded {len(validation):,} validation businesses."
    )

    token_to_queries = build_token_lookup(validation)

    print(
        "Built token lookup with "
        f"{len(token_to_queries):,} distinct tokens."
    )

    source2_ranked = scan_source(
        SOURCE2_FILE,
        validation,
        token_to_queries,
    )

    source3_ranked = scan_source(
        SOURCE3_FILE,
        validation,
        token_to_queries,
    )

    combined_rankings = []

    for index in range(len(validation)):
        combined = (
            source2_ranked[index]
            + source3_ranked[index]
        )

        best_by_id = {}

        for score, candidate_id in combined:
            if (
                candidate_id not in best_by_id
                or score > best_by_id[candidate_id]
            ):
                best_by_id[candidate_id] = score

        ranked = sorted(
            [
                (score, candidate_id)
                for candidate_id, score in best_by_id.items()
            ],
            key=lambda item: item[0],
            reverse=True,
        )

        combined_rankings.append(ranked)

    evaluate_thresholds(
        validation,
        combined_rankings,
    )

    evaluate_shortlists(
        validation,
        combined_rankings,
    )


if __name__ == "__main__":
    main()