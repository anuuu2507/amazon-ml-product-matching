from pathlib import Path
import re
import unicodedata
import difflib
from collections import defaultdict

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dataset" / "train"

SAMPLE_SIZE = 100
CHUNK_SIZE = 50_000

# Pruning limits memory use while scanning millions of training records.
MAX_KEEP_PER_BUSINESS = 300
PRUNE_AT = 600

TOP_K_VALUES = [1, 3, 5]

STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that",
    "inc", "incorporated", "llc", "ltd", "limited",
    "corp", "corporation", "company", "co", "plc",
    "gmbh", "sa", "sas", "bv", "pty", "lp",
    "street", "st", "road", "rd", "avenue", "ave",
    "building", "floor", "suite", "unit", "address",
    "usa", "united", "states"
}


def normalize(value):
    if value is None:
        return ""

    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        ch for ch in value
        if not unicodedata.combining(ch)
    )
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens(value):
    return {
        word
        for word in normalize(value).split()
        if len(word) >= 3 and word not in STOPWORDS
    }


def text_similarity(a, b):
    a = normalize(a)
    b = normalize(b)

    if not a or not b:
        return None

    return difflib.SequenceMatcher(None, a, b).ratio()


def f05(true_ids, predicted_ids):
    if not true_ids and not predicted_ids:
        return 1.0

    if not true_ids or not predicted_ids:
        return 0.0

    correct = len(true_ids & predicted_ids)
    precision = correct / len(predicted_ids)
    recall = correct / len(true_ids)

    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


def read_source1_sample():
    validation_path = TRAIN / "validation_source1.tsv"
    source1_path = TRAIN / "train_source1.tsv"

    validation = pd.read_csv(
        validation_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    source1 = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    # Validation file contains IDs and labels; attach business details.
    sample = validation.merge(
        source1,
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left"
    )

    if sample["business_name"].eq("").any():
        raise ValueError(
            "Some validation IDs could not be joined to Source 1."
        )

    sample = sample.sample(
        n=min(SAMPLE_SIZE, len(sample)),
        random_state=42
    ).reset_index(drop=True)

    return sample


def read_ground_truth():
    gt_path = TRAIN / "train_ground_truth.tsv"

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    return dict(
        zip(
            gt["source1_entity_id"],
            gt["matched_entity_ids"]
        )
    )


def make_query_index(sample):
    """
    Create an inverted token index:
    each token maps to the validation queries that contain it.
    This avoids comparing every training row against all 100 queries.
    """
    queries = {}
    token_to_queries = defaultdict(set)

    for row in sample.itertuples(index=False):
        s1_id = row.source1_entity_id

        name_tokens = tokens(row.business_name)
        address_tokens = tokens(row.business_address)
        all_tokens = name_tokens | address_tokens

        queries[s1_id] = {
            "name": row.business_name,
            "address": row.business_address,
            "country": normalize(row.country),
            "tokens": all_tokens,
            "candidates": {}
        }

        for token in all_tokens:
            token_to_queries[token].add(s1_id)

    return queries, token_to_queries


def add_candidate(query, entity_id, score):
    """
    Keep the best score for each candidate ID.
    Periodically trim oversized candidate dictionaries.
    """
    old_score = query["candidates"].get(entity_id)

    if old_score is None or score > old_score:
        query["candidates"][entity_id] = score

    if len(query["candidates"]) > PRUNE_AT:
        best = sorted(
            query["candidates"].items(),
            key=lambda item: item[1],
            reverse=True
        )[:MAX_KEEP_PER_BUSINESS]

        query["candidates"] = dict(best)


def scan_source(source_num, queries, token_to_queries):
    path = TRAIN / f"train_source{source_num}.tsv"

    print(f"\nScanning {path.name} ...")
    rows_seen = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):
        for row in chunk.itertuples(index=False):
            rows_seen += 1

            row_tokens = (
                tokens(row.business_name)
                | tokens(row.business_address)
            )

            if not row_tokens:
                continue

            # Only consider validation queries sharing at least one token.
            possible_query_ids = set()

            for token in row_tokens:
                possible_query_ids.update(
                    token_to_queries.get(token, ())
                )

            if not possible_query_ids:
                continue

            row_country = normalize(row.country)

            for s1_id in possible_query_ids:
                query = queries[s1_id]

                # Reject known country conflicts.
                if (
                    query["country"]
                    and row_country
                    and query["country"] != row_country
                ):
                    continue

                # Name similarity is the main ranking signal.
                name_sim = text_similarity(
                    query["name"],
                    row.business_name
                )

                if name_sim is None:
                    name_sim = 0.0

                address_sim = text_similarity(
                    query["address"],
                    row.business_address
                )

                if address_sim is None:
                    score = name_sim
                else:
                    score = 0.60 * name_sim + 0.40 * address_sim

                add_candidate(
                    query,
                    row.entity_id,
                    score
                )

        print(f"  Rows scanned: {rows_seen:,}")

    print(f"Finished Source {source_num}.")


def evaluate(queries, truth_map):
    print("\nValidation results")
    print("------------------")

    for k in TOP_K_VALUES:
        business_scores = []
        total_true_ids = 0
        total_retrieved_true_ids = 0
        businesses_with_candidates = 0
        correct_empty_predictions = 0

        for s1_id, query in queries.items():
            ranked = sorted(
                query["candidates"].items(),
                key=lambda item: item[1],
                reverse=True
            )

            predicted = {
                entity_id
                for entity_id, score in ranked[:k]
            }

            raw_truth = truth_map.get(s1_id, "")
            true_ids = {
                item.strip()
                for item in raw_truth.split(",")
                if item.strip()
            }

            business_scores.append(
                f05(true_ids, predicted)
            )

            total_true_ids += len(true_ids)
            total_retrieved_true_ids += len(
                true_ids & predicted
            )

            if predicted:
                businesses_with_candidates += 1

            if not true_ids and not predicted:
                correct_empty_predictions += 1

        mean_f05 = (
            sum(business_scores) / len(business_scores)
            if business_scores else 0.0
        )

        recall = (
            total_retrieved_true_ids / total_true_ids
            if total_true_ids else 0.0
        )

        print(
            f"Top {k}: "
            f"mean per-business F0.5={mean_f05:.4f} | "
            f"matched businesses={businesses_with_candidates}/"
            f"{len(queries)} | "
            f"matched-ID recall={recall:.4f} | "
            f"correct empty={correct_empty_predictions}"
        )


def main():
    print("Loading validation sample and Source 1 details...")
    sample = read_source1_sample()
    truth_map = read_ground_truth()

    print(f"Validation sample size: {len(sample)}")

    queries, token_to_queries = make_query_index(sample)

    print(
        f"Distinct query tokens: {len(token_to_queries):,}"
    )

    scan_source(2, queries, token_to_queries)
    scan_source(3, queries, token_to_queries)

    evaluate(queries, truth_map)


if __name__ == "__main__":
    main()