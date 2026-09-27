"""
Business Entity Resolution Pipeline
====================================
Stage 1: Norm + Blocking (TF-IDF char-ngram per country, cosine sim)
Stage 2: Feature Engineering + LightGBM Classification
Outputs: output/candidate_pairs.tsv, output/matching_results.tsv
"""
import re
import gc
import csv
import sys
import json
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ── Abbreviation map (for name normalisation) ─────────────────────────────────
ABBREV = {
    "corporation": "corp", "incorporated": "inc", "limited": "ltd",
    "private": "pvt", "company": "co", "street": "st", "road": "rd",
    "avenue": "ave", "boulevard": "blvd", "drive": "dr", "lane": "ln",
    "highway": "hwy", "apartment": "apt", "suite": "ste",
    "centre": "center", "and": "&",
}
PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)
SPACE_RE = re.compile(r'\s+')

def normalize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = PUNCT_RE.sub(' ', text)
    tokens = SPACE_RE.split(text.strip())
    tokens = [ABBREV.get(t, t) for t in tokens if t]
    return " ".join(tokens)

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("")
    df["name_norm"]    = df["business_name"].map(normalize)
    df["addr_norm"]    = df["business_address"].map(normalize)
    df["country_norm"] = df["country"].map(normalize)
    df["text"]         = df["name_norm"] + " " + df["addr_norm"]
    return df


# ── Blocking ──────────────────────────────────────────────────────────────────
def generate_candidates(s1: pd.DataFrame, pool: pd.DataFrame,
                         threshold: float = 0.15, top_k: int = 30) -> pd.DataFrame:
    """
    For every Source-1 row find up to `top_k` candidate IDs from `pool`
    within the same country block using char-ngram TF-IDF cosine similarity.
    """
    parts = []
    countries = s1["country_norm"].unique()

    for country in tqdm(countries, desc="Blocking"):
        c_s1   = s1[s1["country_norm"] == country]
        c_pool = pool[pool["country_norm"] == country]
        if len(c_pool) == 0:
            continue

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                              min_df=1, dtype=np.float32, sublinear_tf=True)
        try:
            pool_mat = vec.fit_transform(c_pool["text"])
            s1_mat   = vec.transform(c_s1["text"])
        except ValueError:
            continue

        s1_ids   = c_s1["entity_id"].values
        pool_ids = c_pool["entity_id"].values

        chunk = 5000
        for i in range(0, s1_mat.shape[0], chunk):
            blk = s1_mat[i:i + chunk]
            sim = blk.dot(pool_mat.T).tocsr()

            # Per S1 row: keep top_k above threshold
            for row_i in range(blk.shape[0]):
                row = sim.getrow(row_i)
                d   = row.data
                idx = row.indices
                if len(d) == 0:
                    continue
                mask = d >= threshold
                d, idx = d[mask], idx[mask]
                if len(d) == 0:
                    continue
                if len(d) > top_k:
                    order = np.argpartition(d, -top_k)[-top_k:]
                    d, idx = d[order], idx[order]
                parts.append(
                    pd.DataFrame({
                        "source1_entity_id":  s1_ids[i + row_i],
                        "candidate_entity_id": pool_ids[idx],
                        "tfidf_score":         d.astype(np.float32),
                    })
                )

        del pool_mat, s1_mat
        gc.collect()

    if not parts:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "tfidf_score"])
    return pd.concat(parts, ignore_index=True)


# ── Feature Engineering ───────────────────────────────────────────────────────
def _jaccard_tokens(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    u = sa | sb
    return len(sa & sb) / len(u) if u else 0.0

def _containment(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    d = min(len(sa), len(sb))
    return len(sa & sb) / d if d else 0.0

def _char3_jaccard(a: str, b: str) -> float:
    def tri(s):
        s = s.replace(" ", "")
        return set(s[i:i+3] for i in range(len(s) - 2))
    ta, tb = tri(a), tri(b)
    u = ta | tb
    return len(ta & tb) / len(u) if u else 0.0

def _length_sim(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    return min(la, lb) / max(la, lb) if max(la, lb) > 0 else 1.0

def extract_features(pairs: pd.DataFrame,
                     s1_lookup: dict, cand_lookup: dict) -> np.ndarray:
    """Vectorised-friendly feature extraction."""
    n   = len(pairs)
    out = np.zeros((n, 9), dtype=np.float32)

    s1_ids   = pairs["source1_entity_id"].values
    cand_ids = pairs["candidate_entity_id"].values
    tfidf    = pairs["tfidf_score"].values

    for i in range(n):
        s = s1_lookup.get(s1_ids[i], {})
        c = cand_lookup.get(cand_ids[i], {})

        sn, cn = s.get("name_norm", ""), c.get("name_norm", "")
        sa, ca = s.get("addr_norm", ""), c.get("addr_norm", "")
        sc, cc = s.get("country_norm", ""), c.get("country_norm", "")

        out[i, 0] = tfidf[i]
        out[i, 1] = _jaccard_tokens(sn, cn)
        out[i, 2] = _containment(sn, cn)
        out[i, 3] = _char3_jaccard(sn, cn)
        out[i, 4] = _jaccard_tokens(sa, ca)
        out[i, 5] = _char3_jaccard(sa, ca)
        out[i, 6] = _length_sim(sn, cn)
        out[i, 7] = 1.0 if (sn and sn == cn) else 0.0
        out[i, 8] = 1.0 if (sc and sc == cc) else 0.0

    return out


# ── Scoring ───────────────────────────────────────────────────────────────────
def f05_macro(gold: dict, pred: dict) -> float:
    scores = []
    for s1, truth in gold.items():
        predicted = pred.get(s1, set())
        if not truth:
            scores.append(1.0 if not predicted else 0.0)
            continue
        if not predicted:
            scores.append(0.0)
            continue
        tp        = len(truth & predicted)
        precision = tp / len(predicted)
        recall    = tp / len(truth)
        denom     = 0.25 * precision + recall
        scores.append(1.25 * precision * recall / denom if denom else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ── Write output TSVs ─────────────────────────────────────────────────────────
def write_output(path: Path, s1_ids: pd.Series, id_map: dict,
                 col: str) -> None:
    """Write tab-separated file: source1_entity_id<TAB>comma_list"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\",
                       lineterminator="\n")
        w.writerow(["source1_entity_id", col])
        for sid in s1_ids:
            ids = id_map.get(sid, [])
            # deduplicate preserving order
            seen, deduped = set(), []
            for x in ids:
                if x not in seen:
                    seen.add(x); deduped.append(x)
            w.writerow([sid, ",".join(deduped)])


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(data_dir: Path, out_dir: Path,
        train_sample: int  = 100_000,
        val_frac:     float = 0.20,
        tfidf_thresh: float = 0.15,
        top_k:        int   = 30) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    # ── 1. Load & normalise train data ────────────────────────────────────────
    print("\n[1/7] Loading training data ...")
    s1_full = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str)
    gt_raw  = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str)

    # Build ground-truth lookup
    truth = {}
    for _, row in gt_raw.iterrows():
        sid   = row["source1_entity_id"]
        mids  = str(row.get("matched_entity_ids", "") or "")
        truth[sid] = set(m for m in mids.split(",") if m.strip())

    # Sample S1
    s1_sample = s1_full.sample(min(train_sample, len(s1_full)), random_state=42)
    split     = int(len(s1_sample) * (1 - val_frac))
    s1_tr     = prepare(s1_sample.iloc[:split].copy())
    s1_val    = prepare(s1_sample.iloc[split:].copy())

    print(f"   Train S1: {len(s1_tr):,}   |   Val S1: {len(s1_val):,}")

    print("[2/7] Loading candidate pool (S2+S3) ...")
    t2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", dtype=str)
    t3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", dtype=str)
    cand_pool = prepare(pd.concat([t2, t3], ignore_index=True))
    print(f"   Candidate pool size: {len(cand_pool):,}")

    cand_lookup = (cand_pool.set_index("entity_id")
                             [["name_norm","addr_norm","country_norm"]]
                             .to_dict("index"))

    # ── 2. Blocking (train) ───────────────────────────────────────────────────
    print("[3/7] Generating training candidates ...")
    tr_pairs  = generate_candidates(s1_tr,  cand_pool, tfidf_thresh, top_k)
    val_pairs = generate_candidates(s1_val, cand_pool, tfidf_thresh, top_k)
    print(f"   Train pairs: {len(tr_pairs):,}   |   Val pairs: {len(val_pairs):,}")

    def label_pairs(pairs):
        pairs = pairs.copy()
        pairs["label"] = pairs.apply(
            lambda r: 1 if r["candidate_entity_id"] in truth.get(r["source1_entity_id"], set()) else 0,
            axis=1)
        return pairs

    tr_pairs  = label_pairs(tr_pairs)
    val_pairs = label_pairs(val_pairs)
    print(f"   Train positives: {tr_pairs['label'].sum():,}  "
          f"Val positives: {val_pairs['label'].sum():,}")

    # ── 3. Features ───────────────────────────────────────────────────────────
    print("[4/7] Engineering features ...")
    s1_tr_lookup  = (s1_tr.set_index("entity_id")
                           [["name_norm","addr_norm","country_norm"]]
                           .to_dict("index"))
    s1_val_lookup = (s1_val.set_index("entity_id")
                            [["name_norm","addr_norm","country_norm"]]
                            .to_dict("index"))

    X_tr  = extract_features(tr_pairs,  s1_tr_lookup,  cand_lookup)
    y_tr  = tr_pairs["label"].values.astype(np.int8)
    X_val = extract_features(val_pairs, s1_val_lookup, cand_lookup)
    y_val = val_pairs["label"].values.astype(np.int8)

    # ── 4. Train classifier ───────────────────────────────────────────────────
    print("[5/7] Training LightGBM ...")
    scale_pos = max(1.0, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    clf = lgb.LGBMClassifier(
        n_estimators     = 300,
        learning_rate    = 0.05,
        num_leaves       = 63,
        min_child_samples= 20,
        scale_pos_weight = scale_pos,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = -1,
    )
    clf.fit(X_tr, y_tr,
            eval_set          = [(X_val, y_val)],
            callbacks         = [lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(-1)])

    val_probs = clf.predict_proba(X_val)[:, 1]

    # ── 5. Threshold tuning ───────────────────────────────────────────────────
    gold_val = {sid: truth.get(sid, set()) for sid in s1_val["entity_id"]}

    best_th, best_f = 0.5, -1.0
    for th in np.arange(0.05, 0.96, 0.05):
        val_pairs["pred"] = val_probs >= th
        pred_d = (val_pairs[val_pairs["pred"]]
                  .groupby("source1_entity_id")["candidate_entity_id"]
                  .apply(set).to_dict())
        f = f05_macro(gold_val, pred_d)
        if f > best_f:
            best_f, best_th = f, float(th)

    # Evaluate candidate recall ceiling
    cand_recall_d = (val_pairs.groupby("source1_entity_id")["candidate_entity_id"]
                                .apply(set).to_dict())
    ceiling = f05_macro(gold_val, cand_recall_d)

    # Link recall
    total_gold  = sum(len(v) for v in gold_val.values())
    found_links = sum(len(truth.get(sid, set()) & cand_recall_d.get(sid, set()))
                     for sid in gold_val)
    link_recall = found_links / total_gold if total_gold else 0.0

    print(f"\n{'='*55}")
    print(f"  Val F0.5 (macro)          : {best_f:.4f}")
    print(f"  Candidate ceiling F0.5    : {ceiling:.4f}")
    print(f"  Candidate link recall     : {link_recall:.4f}")
    print(f"  Threshold                 : {best_th:.2f}")
    print(f"{'='*55}\n")

    metrics = {
        "val_f05_macro":          best_f,
        "candidate_ceiling_f05":  ceiling,
        "candidate_link_recall":  link_recall,
        "threshold":              best_th,
        "train_pairs":            int(len(tr_pairs)),
        "val_pairs":              int(len(val_pairs)),
    }
    (out_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    # Free train objects
    del s1_tr, s1_val, cand_pool, tr_pairs, val_pairs, X_tr, X_val, t2, t3
    del cand_lookup, s1_tr_lookup, s1_val_lookup
    gc.collect()

    # ── 6. Test inference ─────────────────────────────────────────────────────
    print("[6/7] Loading test data ...")
    te1 = prepare(pd.read_csv(test_dir / "test_source1.tsv", sep="\t", dtype=str))
    te2 = pd.read_csv(test_dir / "test_source2.tsv", sep="\t", dtype=str)
    te3 = pd.read_csv(test_dir / "test_source3.tsv", sep="\t", dtype=str)
    te_pool = prepare(pd.concat([te2, te3], ignore_index=True))
    print(f"   Test S1: {len(te1):,}  |  Test pool: {len(te_pool):,}")

    te_pool_lookup = (te_pool.set_index("entity_id")
                              [["name_norm","addr_norm","country_norm"]]
                              .to_dict("index"))
    te1_lookup = (te1.set_index("entity_id")
                      [["name_norm","addr_norm","country_norm"]]
                      .to_dict("index"))

    print("[7/7] Generating test candidates & predicting ...")
    te_pairs = generate_candidates(te1, te_pool, tfidf_thresh, top_k)
    print(f"   Test candidate pairs: {len(te_pairs):,}")

    # Build candidate map (for output)
    cand_map: dict[str, list[str]] = {}
    for sid in te1["entity_id"]:
        cand_map[sid] = []
    for _, row in te_pairs.iterrows():
        cand_map.setdefault(row["source1_entity_id"], []).append(row["candidate_entity_id"])

    # Predict in batches
    match_map: dict[str, list[str]] = {sid: [] for sid in te1["entity_id"]}
    if len(te_pairs) > 0:
        X_te     = extract_features(te_pairs, te1_lookup, te_pool_lookup)
        te_probs = clf.predict_proba(X_te)[:, 1]
        te_pairs["prob"] = te_probs
        for _, row in te_pairs.iterrows():
            if row["prob"] >= best_th:
                match_map.setdefault(row["source1_entity_id"], []).append(
                    row["candidate_entity_id"])

    # ── 7. Write outputs ──────────────────────────────────────────────────────
    write_output(out_dir / "candidate_pairs.tsv",
                 te1["entity_id"], cand_map, "candidate_entity_ids")
    write_output(out_dir / "matching_results.tsv",
                 te1["entity_id"], match_map, "matched_entity_ids")

    cands_total  = sum(len(v) for v in cand_map.values())
    matches_total= sum(len(v) for v in match_map.values())
    pair_space   = len(te1) * len(te_pool)
    reduction    = 1.0 - cands_total / pair_space if pair_space else 0.0

    print(f"\n{'='*55}")
    print(f"  Test S1 entities          : {len(te1):,}")
    print(f"  Test candidate pairs      : {cands_total:,}")
    print(f"  Test predicted matches    : {matches_total:,}")
    print(f"  Avg candidates / S1       : {cands_total/max(len(te1),1):.1f}")
    print(f"  Pair-space reduction ratio: {reduction:.6f}")
    print(f"  Outputs written to        : {out_dir.resolve()}")
    print(f"{'='*55}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir",     type=Path, default=Path("dataset"))
    ap.add_argument("--output-dir",   type=Path, default=Path("output"))
    ap.add_argument("--train-sample", type=int,  default=100_000)
    ap.add_argument("--val-frac",     type=float, default=0.20)
    ap.add_argument("--tfidf-thresh", type=float, default=0.15)
    ap.add_argument("--top-k",        type=int,  default=30)
    args = ap.parse_args()
    run(args.data_dir, args.output_dir,
        train_sample  = args.train_sample,
        val_frac      = args.val_frac,
        tfidf_thresh  = args.tfidf_thresh,
        top_k         = args.top_k)

if __name__ == "__main__":
    main()
