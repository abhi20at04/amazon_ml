# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Business Entity Resolution Baseline  
**Team Members:** Not provided  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
This solution uses a deterministic, country-scoped SQLite inverted index to reduce the multi-trillion-pair search space to a bounded shortlist, followed by a trained logistic-regression pair classifier. Name and address similarities drive classification; a seeded held-out sample selects the probability threshold against the challenge's per-entity macro F0.5 metric. The pipeline uses only supplied challenge data and treats countries as open-set strings.

---

## 2. Methodology

### 2.1 Problem Analysis
The training files contain 2,206,821 Source 1 entities and 10,320,219 Source 2/3 records combined; the test files contain 1,732,544 Source 1 entities and 9,969,589 Source 2/3 records combined. Samples show variable casing and punctuation, legal-suffix differences, reordered address components, missing address fields, transliteration/mojibake, and occasionally concatenated city/state/country text. Test includes France, which is absent from training; no country list or country-specific branch is used.

### 2.2 Solution Strategy
Normalize Unicode/diacritics, casing, punctuation, ampersands, and common corporate/address abbreviations. Build a disk-backed target index, retrieve at most 30 records using shared rare country-scoped name/address tokens, engineer pairwise string features, and fit a logistic-regression model from training labels. Select the threshold on the held-out portion of a seeded sample, then stream both TSV outputs for every test Source 1 record.

**Approach Type:** Blocking + classical supervised pair classification  
**Core Innovation:** Rare-token inverted-index retrieval provides a bounded shortlist without materializing all cross-source pairs; SQLite keeps the large target corpus on disk.

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** Normalized open-set country label plus up to three informative name tokens and two address tokens. Candidate ranking sums inverse posting-list frequencies; the top 30 unique target IDs are retained.
- **Candidate pairs generated:** Recorded by the full run in `output/validation_metrics.json` and printed by the pipeline; candidate-generation diagnostics are run-specific.
- **How you ensured true matches were not lost:** Multiple name/address tokens offer alternate routes into the shortlist and the held-out split measures link recall. Blocking is not lossless: links with no shared selected token, inconsistent country labels, or rank below 30 can be missed.

---

## 4. Matching Model

**Features used:**
- Name features: token Jaccard and containment, character-trigram Jaccard, sequence similarity, exact normalized-name match, and relative length similarity.
- Address features: token Jaccard and containment, character-trigram Jaccard, sequence similarity, and exact normalized-address match.
- Other: normalized country equality.

**Model type:** Scikit-learn logistic regression trained from scratch on up to 10,000 sampled Source 1 entities, with up to 15 retrieved negative pairs per training entity. No pretrained model or external business data is used.  
**Threshold selection method:** Maximize per-entity macro F0.5 over the held-out 20% split of a deterministic 12,500-entity reservoir sample.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Reported by the completed pipeline run and persisted in `output/validation_metrics.json` when run with the current source.
- **Common false positives (wrong merges):** Not manually audited; likely risks include businesses sharing generic name/address tokens and short or low-information names.
- **Common false negatives (missed matches):** Not manually audited; blocking may miss records with no selected token overlap, while transliteration, severe typos, or missing fields can reduce pair similarity.

---

## 6. Conclusion
The solution produces a scalable, reproducible shortlist and precision-oriented multi-match predictions using only the supplied data. It supports unseen country labels and records blocking quality and held-out matching metrics so candidate recall limitations and threshold trade-offs remain measurable.

---

## Appendix

### A. Code Artefacts
The runnable entry point is `code/business_entity_resolution/src/pipeline.py`; the README and pinned requirements are in `code/business_entity_resolution/`. Running the documented command rebuilds training and test indexes, selects a threshold, and writes both submission TSVs plus validation diagnostics under `output/`.

### B. Additional Results
The metrics JSON records the validation macro F0.5, oracle blocking ceiling, positive-link recall, threshold, candidate volume, and all-pairs reduction ratio. Exact counts depend on the run settings and are emitted by the pipeline rather than hard-coded here.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
