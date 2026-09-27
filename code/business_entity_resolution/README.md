# Business Entity Resolution Baseline

A self-contained classical pipeline for linking Source 1 businesses to matching Source 2/3 records. It uses only the challenge's supplied training and test TSVs. No external lookups, APIs, databases, pretrained models, or external entity-resolution services are used.

## Requirements

- Python 3.10+ (tested with Python 3.13)
- 8 GB RAM recommended; several GB of free local disk are needed for temporary SQLite indexes and output files
- The pinned packages in `requirements.txt`

From the repository root, install dependencies into the selected environment:

```text
python -m pip install -r code/business_entity_resolution/requirements.txt
```

## Reproduce training, validation, and inference

Run from the repository root so that paths resolve to the challenge data:

```text
python code/business_entity_resolution/src/pipeline.py --data-dir dataset --output-dir output
```

The default reproducible settings use a seeded reservoir sample of 12,500 Source 1 training entities, reserve 20% for validation, and retrieve at most 30 candidates per Source 1 record. Indexing covers all training Source 2/3 records for that sample and all test Source 2/3 records for inference. Both indexes are temporary, disk-backed SQLite databases and are removed automatically when the run finishes.

The command fits a logistic-regression pair classifier from training labels, selects a probability cutoff by macro per-Source-1 F0.5 on the held-out sample, then streams predictions for every test Source 1 row. It writes:

- `output/candidate_pairs.tsv` — the final candidate list scored by the inference model.
- `output/matching_results.tsv` — candidate IDs whose classifier probability meets the validation-selected cutoff.
- `output/validation_metrics.json` — held-out macro F0.5, candidate recall/ceiling, cutoff, candidate-volume, and reduction-ratio diagnostics.

For a faster development run, pass a smaller `--sample-size`; the test candidate index and full test inference are still processed. Candidate breadth is adjustable with `--max-candidates` (default 30).

## Pipeline components

- `src/normalization.py`: Unicode/diacritic handling, lowercase/punctuation normalization, common abbreviation expansion, and generic country-scoped blocking keys.
- `src/blocking.py`: chunked SQLite inverted-index construction and IDF-ranked, bounded candidate retrieval.
- `src/matching.py`: name/address token Jaccard and containment, character trigram Jaccard, sequence similarity, country equality, exact-field, and length similarity features; per-entity F0.5 validation/threshold selection.
- `src/pipeline.py`: deterministic train/validation split, classifier fitting, metrics, and streaming test output generation.

## Blocking strategy and expected reduction

The target-side inverted index is keyed by country label plus up to three informative business-name tokens and two address tokens. For each Source 1 record, matching postings are ranked by inverse posting-list size, and the top 30 unique target IDs form the model's inference shortlist. Country labels are treated as open-set strings: there is no US/India-only enumeration, and France is processed normally. This avoids all-pairs comparisons: roughly 1.73 million Source 1 records × 10 million Source 2/3 records (about 17.3 trillion possible pairs) are replaced by no more than 30 × 1.73 million candidate comparisons. The exact candidate-pair count and reduction ratio are printed after a successful run. Validation also reports candidate-link recall and the resulting candidate-only macro F0.5 ceiling; no recall ceiling is claimed before those values are measured.

This is a recall/precision trade-off: a true link must share at least one selected name/address token under the same country label and rank in the top 30. Records with sparse fields, differing country labels, low-information tokens, or many common-token collisions can be missed. The held-out candidate-link recall is the empirical estimate of this ceiling for the sampled validation set.

## Validation

The pipeline prints held-out macro F0.5, the blocking ceiling and link recall, threshold, test candidate count, mean candidates, and pair-space reduction ratio. Then validate the output format and ID references from the repository root:

```text
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

`--check-ids` loads the multi-million-record target ID sets and can require significant additional RAM. Omit it for the validator's lighter checks; the pipeline itself retrieves candidates directly from the test target index, so emitted IDs come from those test files.

## Model/license

The matcher is a scikit-learn `LogisticRegression` trained from scratch, not a downloaded model or pretrained weights; its fitted coefficients are produced solely from the supplied labels. scikit-learn, NumPy, and SciPy are permissively BSD-licensed implementation dependencies; no third-party model artifact is shipped or loaded. No model artifact is required to reproduce results; training occurs from supplied training files each run.
