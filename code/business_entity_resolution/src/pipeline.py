"""End-to-end train, validate, block, and score business entity records."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from blocking import build_index, read_records, retrieve_candidates
from matching import choose_threshold, macro_fbeta, pair_features
from normalization import normalized_record

SEED = 314159
MAX_CANDIDATES = 30
VALIDATION_SAMPLE = 2500
TRAIN_SAMPLE = 10000
NEGATIVES_PER_SOURCE1 = 15
FEATURE_NAMES = (
    "name_token_jaccard", "name_token_containment", "name_char3_jaccard", "name_sequence_ratio",
    "address_token_jaccard", "address_token_containment", "address_char3_jaccard", "address_sequence_ratio",
    "country_equal", "name_exact", "address_exact", "name_length_similarity",
)


def reservoir_sample(path: Path, size: int, seed: int = SEED) -> list[dict[str, str]]:
    rng = random.Random(seed)
    sample: list[dict[str, str]] = []
    seen = 0
    for row in read_records(path):
        seen += 1
        if len(sample) < size:
            sample.append(row)
        else:
            replacement = rng.randrange(seen)
            if replacement < size:
                sample[replacement] = row
    rng.shuffle(sample)
    return sample


def load_sample_labels(path: Path, selected_ids: set[str]) -> dict[str, set[str]]:
    labels = {entity_id: set() for entity_id in selected_ids}
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            source1_id = (row.get("source1_entity_id") or "").strip()
            if source1_id in labels:
                raw_ids = (row.get("matched_entity_ids") or "").strip()
                labels[source1_id] = {entity_id for entity_id in raw_ids.split(",") if entity_id}
    return labels


def build_training_rows(rows: list[dict[str, str]], labels: dict[str, set[str]], db_path: Path,
                        *, max_candidates: int, negative_limit: int) -> tuple[
                            dict[str, list[tuple[str, list[float]]]],
                            dict[str, set[str]],
                            dict[str, list[str]],
                        ]:
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE pair_features (source_id TEXT, candidate_id TEXT, label INTEGER, features BLOB)")
    validation_scored: dict[str, list[tuple[str, list[float]]]] = {}
    candidate_sets: dict[str, set[str]] = {}
    candidate_lists: dict[str, list[str]] = {}
    split_at = int(len(rows) * 0.8)
    train_rows, validation_rows = rows[:split_at], rows[split_at:]
    try:
        for i, source in enumerate(train_rows):
            source_id = source["entity_id"]
            candidates = retrieve_candidates(db, source, max_candidates=max_candidates)
            feature_source = normalized_record(source)
            positive = []
            negative = []
            for target in candidates:
                pair = (target["entity_id"], pair_features(feature_source, target, normalized=True))
                (positive if target["entity_id"] in labels[source_id] else negative).append(pair)
            db.executemany(
                "INSERT INTO pair_features VALUES (?, ?, ?, ?)",
                [(source_id, candidate_id, 1, np.asarray(vector, dtype=np.float32).tobytes())
                 for candidate_id, vector in positive],
            )
            selected_negative = negative[:negative_limit]
            db.executemany(
                "INSERT INTO pair_features VALUES (?, ?, ?, ?)",
                [(source_id, candidate_id, 0, np.asarray(vector, dtype=np.float32).tobytes())
                 for candidate_id, vector in selected_negative],
            )
            if i % 1000 == 0:
                db.commit()

        db.commit()
        for source in validation_rows:
            source_id = source["entity_id"]
            candidates = retrieve_candidates(db, source, max_candidates=max_candidates)
            feature_source = normalized_record(source)
            candidate_ids = [candidate["entity_id"] for candidate in candidates]
            candidate_lists[source_id] = candidate_ids
            candidate_sets[source_id] = set(candidate_ids)
            validation_scored[source_id] = [
                (candidate["entity_id"], pair_features(feature_source, candidate, normalized=True))
                for candidate in candidates
            ]
    finally:
        db.close()
    return validation_scored, candidate_sets, candidate_lists


def fit_model(db_path: Path, feature_count: int = len(FEATURE_NAMES)) -> tuple[LogisticRegression, int]:
    """Fit on disk-held compact vectors without retaining all training pairs in RAM."""
    db = sqlite3.connect(db_path)
    x_batches = []
    y_batches = []
    try:
        model = LogisticRegression(max_iter=300, solver="liblinear", random_state=SEED)
        cursor = db.execute("SELECT label, features FROM pair_features ORDER BY rowid")
        batch_size = 25000
        pair_count = 0
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            y_batch = np.fromiter((row[0] for row in rows), dtype=np.int8, count=len(rows))
            x_batch = np.vstack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
            x_batches.append(x_batch)
            y_batches.append(y_batch)
            pair_count += len(rows)
        if not x_batches or len(np.unique(np.concatenate(y_batches))) < 2:
            raise RuntimeError("Training sample did not produce both positive and negative candidate pairs.")
        model.fit(np.vstack(x_batches), np.concatenate(y_batches))
        return model, pair_count
    finally:
        db.close()


def _write_stream_row(writer: csv.writer, source1_id: str, ids: list[str]) -> None:
    writer.writerow((source1_id, ",".join(dict.fromkeys(ids))))


def run(data_dir: Path, output_dir: Path, *, sample_size: int = TRAIN_SAMPLE + VALIDATION_SAMPLE,
        max_candidates: int = MAX_CANDIDATES) -> dict[str, float]:
    train_dir, test_dir = data_dir / "train", data_dir / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_s1 = train_dir / "train_source1.tsv"
    targets_train = [train_dir / "train_source2.tsv", train_dir / "train_source3.tsv"]
    test_s1 = test_dir / "test_source1.tsv"
    targets_test = [test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"]

    sampled_rows = reservoir_sample(train_s1, sample_size)
    selected_ids = {row["entity_id"] for row in sampled_rows}
    labels = load_sample_labels(train_dir / "train_ground_truth.tsv", selected_ids)
    metrics: dict[str, float] = {}

    with tempfile.TemporaryDirectory(prefix="entity-resolution-") as work:
        index_path = Path(work) / "train.sqlite"
        indexed_targets = build_index(targets_train, index_path)
        scored_features, candidate_sets, candidate_lists = build_training_rows(
            sampled_rows, labels, index_path,
            max_candidates=max_candidates, negative_limit=NEGATIVES_PER_SOURCE1,
        )
        model, train_pair_count = fit_model(index_path)
        gold_validation = {row["entity_id"]: labels[row["entity_id"]]
                           for row in sampled_rows[int(len(sampled_rows) * 0.8):]}
        scored_probabilities = {}
        for source_id, pairs in scored_features.items():
            if not pairs:
                scored_probabilities[source_id] = []
                continue
            probabilities = model.predict_proba(
                np.asarray([vector for _, vector in pairs], dtype=np.float32)
            )[:, 1]
            scored_probabilities[source_id] = [
                (candidate_id, float(probability))
                for (candidate_id, _), probability in zip(pairs, probabilities)
            ]
        threshold, validation_score = choose_threshold(gold_validation, scored_probabilities)
        val_candidates = {source_id: candidate_sets.get(source_id, set()) for source_id in gold_validation}
        # Oracle classification can reject all negatives but cannot recover links
        # omitted by blocking; score exactly the retrieved gold links for that ceiling.
        oracle_retrieved = {
            source_id: gold_validation[source_id] & val_candidates.get(source_id, set())
            for source_id in gold_validation
        }
        candidate_ceiling = macro_fbeta(gold_validation, oracle_retrieved)
        gold_links = sum(map(len, gold_validation.values()))
        found_links = sum(len(gold & candidate_sets.get(source_id, set()))
                          for source_id, gold in gold_validation.items())
        validation_links_recall = found_links / gold_links if gold_links else 0.0
        average_candidates = sum(map(len, candidate_sets.values())) / max(len(candidate_sets), 1)
        metrics.update({
            "sampled_train_entities": float(len(sampled_rows)),
            "training_pairs": float(train_pair_count),
            "validation_entities": float(len(gold_validation)),
            "validation_f0_5_macro": float(validation_score),
            "validation_blocking_macro_ceiling": float(candidate_ceiling),
            "validation_candidate_link_recall": float(validation_links_recall),
            "validation_mean_candidates": float(average_candidates),
            "probability_threshold": float(threshold),
        })
        print(f"Held-out macro F0.5: {validation_score:.6f} (threshold={threshold:.6f}, "
              f"blocking ceiling={candidate_ceiling:.6f}, link recall={validation_links_recall:.6f})",
              flush=True)
        with open(output_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            f.write("\n")
        del scored_features, candidate_sets, candidate_lists
        index_path.unlink(missing_ok=True)

        test_index = Path(work) / "test.sqlite"
        indexed_test_targets = build_index(targets_test, test_index)
        db = sqlite3.connect(test_index)
        candidate_path = output_dir / "candidate_pairs.tsv"
        matching_path = output_dir / "matching_results.tsv"
        test_count = candidate_count = match_count = 0
        try:
            with open(candidate_path, "w", encoding="utf-8", newline="") as cf, \
                    open(matching_path, "w", encoding="utf-8", newline="") as mf:
                candidate_writer = csv.writer(cf, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
                matching_writer = csv.writer(mf, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
                candidate_writer.writerow(("source1_entity_id", "candidate_entity_ids"))
                matching_writer.writerow(("source1_entity_id", "matched_entity_ids"))
                for source in read_records(test_s1):
                    test_count += 1
                    candidates = retrieve_candidates(db, source, max_candidates=max_candidates)
                    candidate_ids = [candidate["entity_id"] for candidate in candidates]
                    candidate_count += len(candidate_ids)
                    _write_stream_row(candidate_writer, source["entity_id"], candidate_ids)
                    if candidates:
                        feature_source = normalized_record(source)
                        matrix = np.asarray([
                            pair_features(feature_source, candidate, normalized=True)
                            for candidate in candidates
                        ], dtype=np.float32)
                        probabilities = model.predict_proba(matrix)[:, 1]
                        matched = [candidate_ids[i] for i, probability in enumerate(probabilities) if probability >= threshold]
                    else:
                        matched = []
                    match_count += len(matched)
                    _write_stream_row(matching_writer, source["entity_id"], matched)
                    if test_count % 100000 == 0:
                        print(f"Scored {test_count:,} test entities; {candidate_count:,} candidates so far...", flush=True)
        finally:
            db.close()

    pair_space = test_count * indexed_test_targets
    metrics.update({
        "sampled_train_entities": float(len(sampled_rows)),
        "training_pairs": float(train_pair_count),
        "validation_entities": float(len(gold_validation)),
        "validation_f0_5_macro": float(validation_score),
        "validation_blocking_macro_ceiling": float(candidate_ceiling),
        "validation_candidate_link_recall": float(validation_links_recall),
        "validation_mean_candidates": float(average_candidates),
        "probability_threshold": float(threshold),
        "test_source1_entities": float(test_count),
        "test_target_entities": float(indexed_test_targets),
        "test_candidate_pairs": float(candidate_count),
        "test_predicted_matches": float(match_count),
        "test_mean_candidates": candidate_count / max(test_count, 1),
        "test_reduction_ratio": 1.0 - candidate_count / max(pair_space, 1),
    })
    with open(output_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--sample-size", type=int, default=TRAIN_SAMPLE + VALIDATION_SAMPLE)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    args = parser.parse_args()
    metrics = run(args.data_dir, args.output_dir, sample_size=args.sample_size,
                  max_candidates=args.max_candidates)
    print("Validation and inference metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
