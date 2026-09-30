import re
import time
from collections import Counter, defaultdict

import pandas as pd


# Run from the project root:
# py src\validate_matcher.py

ROOT = "dataset"
CHUNK_SIZE = 100_000
FREQUENCY_CUTOFF = 50_000
SAMPLE_SIZE = 100

THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TOP_K_VALUES = [1, 3, 5, 10, 20]


def read_tsv(path, usecols=None):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=usecols,
    )


def tokenize(value):
    """Convert text to lowercase letter/number tokens."""
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def get_text_tokens(row):
    name_tokens = tokenize(row["business_name"])
    address_tokens = tokenize(row["business_address"])
    return name_tokens, address_tokens


def jaccard(a, b):
    if not a and not b:
        return 0.0

    union = a | b
    if not union:
        return 0.0

    return len(a & b) / len(union)


def get_score(query_name, query_address, candidate_name, candidate_address):
    """
    Compare name and address separately.

    When both addresses have tokens, combine name and address scores.
    When an address is missing, lower confidence in the name-only score.
    """
    name_score = jaccard(query_name, candidate_name)

    if query_address and candidate_address:
        address_score = jaccard(query_address, candidate_address)
        return 0.6 * name_score + 0.4 * address_score

    return 0.65 * name_score


def load_validation_sample():
    print("Loading validation IDs and true matches...")

    # Validation file contains IDs and labels, not business details.
    validation_labels = read_tsv(
        f"{ROOT}/train/validation_source1.tsv",
        usecols=["source1_entity_id", "matched_entity_ids"],
    )

    validation_labels = validation_labels.sample(
        n=min(SAMPLE_SIZE, len(validation_labels)),
        random_state=42,
    ).reset_index(drop=True)

    query_ids = set(validation_labels["source1_entity_id"])

    print("Finding sampled business details in train_source1.tsv...")
    details = []

    for chunk in pd.read_csv(
        f"{ROOT}/train/train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        chunksize=CHUNK_SIZE,
    ):
        found = chunk[chunk["entity_id"].isin(query_ids)]
        if not found.empty:
            details.append(found)

    if not details:
        raise ValueError(
            "Could not find sampled validation IDs in train_source1.tsv"
        )

    validation_details = pd.concat(details, ignore_index=True)
    validation_details = validation_details.rename(
        columns={"entity_id": "source1_entity_id"}
    )

    validation = validation_labels.merge(
        validation_details,
        on="source1_entity_id",
        how="inner",
        validate="one_to_one",
    )

    if len(validation) != len(validation_labels):
        missing_count = len(validation_labels) - len(validation)
        raise ValueError(
            f"Could not find details for {missing_count} sampled IDs."
        )

    print(f"Validation businesses loaded: {len(validation)}")
    return validation


def main():
    start = time.time()

    validation = load_validation_sample()

    queries = {}

    for _, row in validation.iterrows():
        name_tokens, address_tokens = get_text_tokens(row)

        queries[row["source1_entity_id"]] = {
            "name": name_tokens,
            "address": address_tokens,
            "country": row["country"].strip().lower(),
        }

    query_ids = set(queries.keys())

    truth_lookup = dict(
        zip(
            validation["source1_entity_id"],
            validation["matched_entity_ids"],
        )
    )

    print("\nPass 1/2: counting token frequencies...")
    token_frequency = Counter()

    for source_name in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"{ROOT}/train/{source_name}"
        print(f"  Counting {source_name}...")

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=["business_name", "business_address"],
            chunksize=CHUNK_SIZE,
        ):
            for _, row in chunk.iterrows():
                tokens = tokenize(row["business_name"])
                tokens.update(tokenize(row["business_address"]))
                token_frequency.update(tokens)

    # Ignore very common tokens to reduce noisy candidate generation.
    useful_tokens = {
        token
        for token, count in token_frequency.items()
        if count < FREQUENCY_CUTOFF
    }

    # Filter query and candidate tokens using the same vocabulary.
    for query in queries.values():
        query["name"] &= useful_tokens
        query["address"] &= useful_tokens
        query["all"] = query["name"] | query["address"]

    # Inverted index: useful token -> validation query IDs.
    token_to_queries = defaultdict(set)

    for query_id, query in queries.items():
        for token in query["all"]:
            token_to_queries[token].add(query_id)

    print(f"Useful tokens: {len(useful_tokens):,}")
    print("\nPass 2/2: scoring candidate pairs...")

    # Store best score for each validation query and candidate ID.
    scores_by_query = defaultdict(dict)

    for source_name in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"{ROOT}/train/{source_name}"
        print(f"  Scoring {source_name}...")

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ],
            chunksize=CHUNK_SIZE,
        ):
            for _, row in chunk.iterrows():
                candidate_id = row["entity_id"]
                candidate_country = row["country"].strip().lower()

                candidate_name, candidate_address = get_text_tokens(row)
                candidate_name &= useful_tokens
                candidate_address &= useful_tokens
                candidate_all = candidate_name | candidate_address

                if not candidate_all:
                    continue

                # Only consider queries sharing at least one useful token.
                possible_queries = set()

                for token in candidate_all:
                    possible_queries.update(
                        token_to_queries.get(token, ())
                    )

                for query_id in possible_queries:
                    query = queries[query_id]

                    # Skip only if both country values exist and disagree.
                    if (
                        query["country"]
                        and candidate_country
                        and query["country"] != candidate_country
                    ):
                        continue

                    score = get_score(
                        query["name"],
                        query["address"],
                        candidate_name,
                        candidate_address,
                    )

                    if score <= 0:
                        continue

                    previous_score = scores_by_query[query_id].get(
                        candidate_id,
                        0.0,
                    )

                    if score > previous_score:
                        scores_by_query[query_id][candidate_id] = score

    # Parse comma-separated ground-truth IDs into sets.
    true_by_query = {}

    for query_id in query_ids:
        raw = truth_lookup.get(query_id, "")
        true_by_query[query_id] = {
            value.strip()
            for value in raw.split(",")
            if value.strip()
        }

    total_true_ids = sum(
        len(ids) for ids in true_by_query.values()
    )

    singleton_ids = {
        query_id
        for query_id, true_ids in true_by_query.items()
        if not true_ids
    }

    print("\nValidation results: threshold + top-K")
    print(f"Sample businesses: {len(query_ids)}")
    print(f"True matched IDs: {total_true_ids}")
    print(f"Businesses with no true matches: {len(singleton_ids)}")

    print(
        "\nTop-K | Threshold | Precision | Recall | F0.5  | "
        "Predicted IDs | Correct empty"
    )
    print("-" * 88)

    # Rank each query's candidate IDs by score, highest first.
    ranked_by_query = {}

    for query_id in query_ids:
        ranked_by_query[query_id] = sorted(
            scores_by_query.get(query_id, {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )

    for top_k in TOP_K_VALUES:
        for threshold in THRESHOLDS:
            true_positives = 0
            total_predicted = 0
            correct_empty = 0

            for query_id in query_ids:
                true_ids = true_by_query[query_id]

                # Keep only the highest-scoring candidates above threshold.
                ranked = ranked_by_query[query_id]
                predicted_ids = {
                    candidate_id
                    for candidate_id, score in ranked[:top_k]
                    if score >= threshold
                }

                true_positives += len(predicted_ids & true_ids)
                total_predicted += len(predicted_ids)

                if query_id in singleton_ids and not predicted_ids:
                    correct_empty += 1

            precision = (
                true_positives / total_predicted
                if total_predicted
                else 0.0
            )

            recall = (
                true_positives / total_true_ids
                if total_true_ids
                else 0.0
            )

            if precision + recall:
                f05 = (
                    1.25 * precision * recall
                    / (0.25 * precision + recall)
                )
            else:
                f05 = 0.0

            print(
                f"{top_k:5d} | "
                f"{threshold:9.2f} | "
                f"{precision:9.4f} | "
                f"{recall:6.4f} | "
                f"{f05:5.4f} | "
                f"{total_predicted:13,} | "
                f"{correct_empty:13,}"
            )

    elapsed_minutes = (time.time() - start) / 60
    print(f"\nFinished in {elapsed_minutes:.1f} minutes.")
    print(
        "Note: These are diagnostic pair-level metrics on a small sample, "
        "not a guaranteed leaderboard score."
    )


if __name__ == "__main__":
    main()