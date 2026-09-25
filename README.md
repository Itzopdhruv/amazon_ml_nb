# Amazon ML Challenge 2026: Business Entity Resolution

Match each Source-1 business record to all of its duplicates in Source-2 and Source-3. Scoring is
macro-averaged per-entity F0.5, with singletons included.

Everything is in one Colab notebook:
[`amazon_ml_entity_resolution_pipeline_colab.ipynb`](amazon_ml_entity_resolution_pipeline_colab.ipynb)

## Pipeline (full data, no dev mode)

1. **Normalize.** Unicode-safe (keeps Devanagari combining marks), legal suffixes and street
   abbreviations canonicalized. Each record becomes one text: `name: … | address: …`.
2. **Embed.** Frozen `BAAI/bge-m3` (no fine-tuning), fp16, 1024-d CLS vectors for every train and test
   record. Stored on Google Drive as 250k-row shards (~26 GB for train).
3. **Block.** Exact GPU inner-product search, **hard-filtered by country**: top-25 from S2 plus top-25
   from S3 per S1 record. The best 20 of those go to the reranker, and that set is written as
   `candidate_pairs.tsv`.
4. **Rerank.** Fine-tuned `BAAI/bge-reranker-v2-m3` cross-encoder, trained on ~1.7M pairs from 150k train
   entities (every ground-truth positive, plus 4 hardest and 4 random retrieved negatives each).
5. **Decide.** Two thresholds tuned for macro F0.5 on 50k held-out validation entities. Predict every
   candidate scoring at least `t_all`; if none does, predict the top candidate when it scores at least
   `t_top`.
6. **Submit.** Writes `submission/matching_results.tsv` and `submission/candidate_pairs.tsv` and checks
   them against every rule in the problem statement.

## Running it

1. Colab: Runtime → Change runtime type → **L4 GPU** (T4 works too, just slower).
2. Train TSVs go in `MyDrive/amazon_ml_challenge/raw_data/`. If they aren't there, the notebook
   downloads them from Kaggle using the `KAGGLE_USERNAME` / `KAGGLE_KEY` Colab secrets.
3. Test TSVs go in `MyDrive/amazon_ml_challenge/raw_data/test/`.
4. **Runtime → Run all.** All outputs go to `MyDrive/amazon_ml_challenge/full_v2/`. Every stage skips
   itself if its output already exists, so after a disconnect just **Run all** again. Embedding and
   scoring resume from the last saved shard or chunk. Training resumes from the last Drive checkpoint,
   saved every 30 minutes.

Rough time on an L4: embeddings ~2 h for the 12.5M train records (plus the test set), cross-encoder
training a few hours, retrieval minutes, validation scoring ~15 min. Each stage logs its own ETA.

Knobs are all in Cell 2. The ones worth touching are `N_CE_TRAIN_ENTITIES` (more training data means
longer training), `RERANK_K` (candidates per entity), and `K_PER_SOURCE`.

## Next steps

- [ ] Full run: record recall@K (Cell 9) and validation F0.5 (Cell 12) here
- [ ] First leaderboard submission
- [ ] If recall@20 is the bottleneck: raise `RERANK_K`, or add a lexical (char n-gram TF-IDF) retriever
      alongside bge-m3
- [ ] If there's GPU budget left: raise `N_CE_TRAIN_ENTITIES`, or add a second epoch
- [ ] Final package (`code/src`, `requirements.txt`, methodology doc) per the problem statement

Raw data, embeddings and checkpoints are git-ignored. They live on Drive, not in this repo.
