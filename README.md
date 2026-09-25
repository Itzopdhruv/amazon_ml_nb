# Amazon ML Challenge 2026 — Business Entity Resolution

Match each Source-1 business record to all of its duplicates in Source-2 / Source-3, scored by
macro-averaged per-entity F0.5 (singletons count: predicting nothing for a true singleton scores 1.0).

Everything lives in one Colab notebook:
[`amazon_ml_entity_resolution_pipeline_colab.ipynb`](amazon_ml_entity_resolution_pipeline_colab.ipynb)

## Pipeline

1. **Normalize**: Unicode-safe punctuation stripping (keeps Indic combining marks), plus legal-suffix
   and street-abbreviation canonicalization.
2. **Block (Stage 1)**: embed with `BAAI/bge-m3` (optionally fine-tuned on GT pairs), use a per-country
   FAISS index (IVF-PQ for big countries, exact flat index for small ones), and take the top-50 S2 plus
   top-50 S3 hits per S1 record.
3. **Prefilter**: keep the top-15 by bi-encoder score. This is the official `candidate_pairs.tsv`.
4. **Rerank (Stage 2)**: a LightGBM model on similarity features (default), or a fine-tuned
   `bge-reranker-v2-m3` cross-encoder, trained on hard negatives mined in Stage 1.
5. **Decide (Stage 3)**: grid-search the score threshold for macro F0.5 on a held-out 5% of S1, then
   apply it to the test set. This writes `matching_results.tsv`.

All expensive stages are resumable and synced to Google Drive, so a Colab disconnect only costs the
stage that was running.

## How to run

1. Colab → Runtime → T4 GPU. Run top to bottom with `DEV_MODE = True` first (takes minutes).
2. Compare recall@k (Cell 12) with `SKIP_BIENCODER_FINETUNE = True` vs `False`. Each setting gets its
   own run-tagged folders (`*_dev_offshelf` vs `*_dev`), so both results stay cached.
3. Set `DEV_MODE = False` for the full run. It spans several sessions, and `restore_from_drive()` picks up
   where the last one stopped.
4. Put `test_source{1,2,3}.tsv` in `MyDrive/amazon_ml_challenge/test_data/`, set
   `RUN_TEST_INFERENCE = True` (Section 19), and collect `submission/matching_results.tsv` and
   `submission/candidate_pairs.tsv` from Drive.

## Status / next steps

- [x] End-to-end pipeline, validation F0.5, test-set inference cell
- [ ] DEV_MODE run on real data: record recall@k (off-the-shelf vs fine-tuned) and val F0.5 here
- [ ] Full run + first submission
- [ ] Cheap GBDT wins to try next: per-source features (S2 vs S3 behave differently), candidate rank /
      score gap to the top-1 hit, token-level fuzzy name match (e.g. `rapidfuzz` token_set_ratio),
      name-only vs address-only bi-encoder cosine
- [ ] Per-entity decision rules: e.g. always keep the top-1 candidate if it clears a lower threshold.
      Missing every match on a non-singleton scores 0, so this is usually worth it under F0.5
- [ ] Cross-encoder reranker (`RERANKER_APPROACH = "cross_encoder"`) if GPU budget allows

Raw data, indexes and checkpoints are git-ignored. They live in Drive and Kaggle, not in this repo.
