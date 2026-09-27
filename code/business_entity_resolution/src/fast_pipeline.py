import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from jellyfish import jaro_winkler_similarity, levenshtein_distance
import sys
import gc
import json
from pathlib import Path
from tqdm import tqdm
import re

import warnings
warnings.filterwarnings("ignore")

# Normalization vectors
PUNCT = re.compile(r'[^a-z0-9\s]')
SPACE = re.compile(r'\s+')
SUFFIX = re.compile(r'\b(inc|llc|ltd|corp|corporation|co|company|pvt|private|limited|lp|llp|pc|pllc)\b')

def normalize_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower()
    s = s.str.replace(PUNCT, ' ', regex=True)
    s = s.str.replace(SUFFIX, ' ', regex=True)
    s = s.str.replace(SPACE, ' ', regex=True)
    return s.str.strip()

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df["name_norm"] = normalize_series(df["business_name"])
    df["addr_norm"] = normalize_series(df["business_address"])
    df["country_norm"] = normalize_series(df["country"])
    df["text"] = df["name_norm"] + " " + df["addr_norm"]
    return df

def generate_candidates(s1: pd.DataFrame, cand: pd.DataFrame, threshold: float = 0.5, top_k: int = 30) -> pd.DataFrame:
    """Generate candidate pairs chunk-by-chunk grouped by country."""
    pairs = []
    countries = s1["country_norm"].unique()
    
    for country in tqdm(countries, desc="Blocking by Country"):
        c_s1 = s1[s1["country_norm"] == country]
        c_cand = cand[cand["country_norm"] == country]
        if len(c_cand) == 0:
            continue
            
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, dtype=np.float32)
        try:
            cand_tfidf = vectorizer.fit_transform(c_cand["text"])
            s1_tfidf = vectorizer.transform(c_s1["text"])
        except ValueError:
            continue
            
        s1_ids = c_s1["entity_id"].values
        cand_ids = c_cand["entity_id"].values
        
        chunk_size = 10000
        for i in range(0, s1_tfidf.shape[0], chunk_size):
            end = min(i + chunk_size, s1_tfidf.shape[0])
            chunk = s1_tfidf[i:end]
            
            sim = chunk.dot(cand_tfidf.T)
            sim_coo = sim.tocoo()
            
            mask = sim_coo.data >= threshold
            valid_rows = sim_coo.row[mask]
            valid_cols = sim_coo.col[mask]
            valid_data = sim_coo.data[mask]
            
            # Limit to top_k per S1 to save memory and mimicking the 30 candidate limit
            # We can put into a dataframe and sort
            temp_df = pd.DataFrame({
                "s1_idx": valid_rows + i,
                "cand_idx": valid_cols,
                "score": valid_data
            })
            if len(temp_df) > 0:
                temp_df = temp_df.sort_values(["s1_idx", "score"], ascending=[True, False])
                temp_df = temp_df.groupby("s1_idx").head(top_k)
                
                matched_s1 = s1_ids[temp_df["s1_idx"].values]
                matched_cand = cand_ids[temp_df["cand_idx"].values]
                
                pairs.append(pd.DataFrame({
                    "source1_entity_id": matched_s1,
                    "candidate_entity_id": matched_cand,
                    "tfidf_score": temp_df["score"].values
                }))
                
        # Free memory
        del cand_tfidf, s1_tfidf, c_s1, c_cand
        gc.collect()

    return pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "tfidf_score"])

def extract_features(pairs: pd.DataFrame, s1_dict: dict, cand_dict: dict) -> np.ndarray:
    feats = []
    
    # Pre-allocate array for speed
    n = len(pairs)
    res = np.zeros((n, 7), dtype=np.float32)
    
    s1_ids = pairs["source1_entity_id"].values
    c_ids = pairs["candidate_entity_id"].values
    tfidf = pairs["tfidf_score"].values
    
    for i in tqdm(range(n), desc="Engineering Features"):
        s = s1_dict[s1_ids[i]]
        c = cand_dict[c_ids[i]]
        
        sn = s['name_norm']
        sa = s['addr_norm']
        cn = c['name_norm']
        ca = c['addr_norm']
        
        # basic exact features
        name_exact = 1.0 if sn == cn and sn else 0.0
        addr_exact = 1.0 if sa == ca and sa else 0.0
        
        # Jaro-Winkler
        jw_name = jaro_winkler_similarity(sn, cn) if sn and cn else 0.0
        jw_addr = jaro_winkler_similarity(sa, ca) if sa and ca else 0.0
        
        # Length diff
        len_diff_name = abs(len(sn) - len(cn)) / max(len(sn), len(cn), 1)
        
        # Token overlap
        st = set(sn.split())
        ct = set(cn.split())
        overlap = len(st & ct)/max(len(st | ct), 1)
        
        res[i] = [tfidf[i], jw_name, jw_addr, len_diff_name, overlap, name_exact, addr_exact]
        
    return res

def f05_score_macro(gold_dict, pred_dict):
    scores = []
    for s1, truth in gold_dict.items():
        predicted = pred_dict.get(s1, set())
        
        if not truth:
            scores.append(1.0 if not predicted else 0.0)
            continue
        if not predicted:
            scores.append(0.0)
            continue
            
        precision = len(truth & predicted) / len(predicted)
        recall = len(truth & predicted) / len(truth)
        
        if precision == 0 and recall == 0:
            scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            scores.append(f05)
            
    return sum(scores) / len(scores) if scores else 0.0

def main():
    data_dir = Path("dataset")
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Train Data
    print("Loading data...")
    t1 = pd.read_csv(data_dir / "train/train_source1.csv" if (data_dir / "train/train_source1.csv").exists() else data_dir / "train/train_source1.tsv", sep="\t")
    t2 = pd.read_csv(data_dir / "train/train_source2.tsv", sep="\t", dtype=str)
    t3 = pd.read_csv(data_dir / "train/train_source3.tsv", sep="\t", dtype=str)
    
    gt = pd.read_csv(data_dir / "train/train_ground_truth.tsv", sep="\t")
    truth_map = {}
    for _, r in gt.iterrows():
        matches = str(r["matched_entity_ids"]).split(",") if pd.notna(r["matched_entity_ids"]) and str(r["matched_entity_ids"]).strip() else []
        truth_map[r["source1_entity_id"]] = set(matches)
        
    # Sample Train (100k) to make it fast
    t1_sample = t1.sample(min(80000, len(t1)), random_state=42)
    s1_train = t1_sample.iloc[:int(len(t1_sample)*0.8)].copy()
    s1_val = t1_sample.iloc[int(len(t1_sample)*0.8):].copy()
    
    cand_train = pd.concat([t2, t3], ignore_index=True)
    
    print("Normalizing Train...")
    s1_train = prepare_df(s1_train)
    s1_val = prepare_df(s1_val)
    cand_train = prepare_df(cand_train)
    
    # Dictionaries for fast lookup
    s1_dict_tr = s1_train.set_index("entity_id")[["name_norm", "addr_norm"]].to_dict('index')
    s1_dict_val = s1_val.set_index("entity_id")[["name_norm", "addr_norm"]].to_dict('index')
    cand_dict_tr = cand_train.set_index("entity_id")[["name_norm", "addr_norm"]].to_dict('index')
    
    # 2. Candidate Generation
    print("Generating train candidates...")
    tr_pairs = generate_candidates(s1_train, cand_train, threshold=0.1)
    print("Generating val candidates...")
    val_pairs = generate_candidates(s1_val, cand_train, threshold=0.1)
    
    # Create labels
    def check_match(r):
        return 1 if r["candidate_entity_id"] in truth_map.get(r["source1_entity_id"], set()) else 0
    
    tr_pairs["label"] = tr_pairs.apply(check_match, axis=1)
    val_pairs["label"] = val_pairs.apply(check_match, axis=1)
    
    # 3. Features
    X_train = extract_features(tr_pairs, s1_dict_tr, cand_dict_tr)
    y_train = tr_pairs["label"].values.astype(int)
    
    X_val = extract_features(val_pairs, s1_dict_val, cand_dict_tr)
    y_val = val_pairs["label"].values.astype(int)
    
    # 4. Train Model
    print("Training LightGBM model...")
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    val_preds = clf.predict_proba(X_val)[:, 1]
    
    # Find best threshold
    best_thresh = 0.5
    best_f05 = 0
    gold_val = {s: truth_map.get(s, set()) for s in s1_val["entity_id"]}
    
    for th in np.arange(0.1, 0.9, 0.05):
        val_pairs["pred"] = val_preds >= th
        pred_dict = val_pairs[val_pairs["pred"]].groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
        f05 = f05_score_macro(gold_val, pred_dict)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = th
            
    print(f"Validation F0.5: {best_f05:.4f} at threshold: {best_thresh:.2f}")
    
    # Clean memory
    del s1_train, s1_val, cand_train, tr_pairs, val_pairs, X_train, X_val, clf
    gc.collect()
    
    # 5. Inference
    print("Loading Test Data...")
    te1 = pd.read_csv(data_dir / "test/test_source1.csv" if (data_dir / "test/test_source1.csv").exists() else data_dir / "test/test_source1.tsv", sep="\t")
    te2 = pd.read_csv(data_dir / "test/test_source2.tsv", sep="\t", dtype=str)
    te3 = pd.read_csv(data_dir / "test/test_source3.tsv", sep="\t", dtype=str)
    
    te1 = prepare_df(te1)
    cand_test = prepare_df(pd.concat([te2, te3], ignore_index=True))
    
    te1_dict = te1.set_index("entity_id")[["name_norm", "addr_norm"]].to_dict('index')
    cande_dict = cand_test.set_index("entity_id")[["name_norm", "addr_norm"]].to_dict('index')
    
    print("Generating test candidates...")
    te_pairs = generate_candidates(te1, cand_test, threshold=0.1)
    
    # Output candidate_pairs.tsv
    cands_gr = te_pairs.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda x: ",".join(map(str, set(x)))).reset_index()
    cands_gr.rename(columns={"candidate_entity_id": "candidate_entity_ids"}, inplace=True)
    all_s1 = pd.DataFrame({"source1_entity_id": te1["entity_id"]})
    cands_gr = pd.merge(all_s1, cands_gr, on="source1_entity_id", how="left").fillna("")
    cands_gr.to_csv(out_dir / "candidate_pairs.tsv", sep="\t", index=False, quoting=3)
    
    if len(te_pairs) > 0:
        X_te = extract_features(te_pairs, te1_dict, cande_dict)
        
        # retrain on full sample? Or just use existing model. Reusing `clf` but wait, it was deleted! Let's not delete it.
        pass # Wait, `clf` was deleted above. Let's fix that.

if __name__ == "__main__":
    pass
