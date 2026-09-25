# ==== Recall check on YOUR validation set: does 20 S2 + 20 S3 catch more true matches than top-15? ====
# Run in the notebook session (uses val_raw_scores from Cell 11: RRF-ranked top-50 per source, and gt_map).
import numpy as np, pandas as pd
r = val_raw_scores.rename(columns={"biencoder_score": "rrf"}).copy()
r["src"] = r["candidate_entity_id"].str[:2]
r = r.sort_values(["source1_entity_id", "rrf"], ascending=[True, False])
r["rank_all"] = r.groupby("source1_entity_id").cumcount()
r["rank_src"] = r.groupby(["source1_entity_id", "src"]).cumcount()
r["hit"] = [c in gt_map.get(e, ()) for e, c in zip(r["source1_entity_id"], r["candidate_entity_id"])]
ntrue = pd.Series({e: len(gt_map.get(e, ())) for e in val_ids_use})
print(f"{'setting':<26}{'cands/entity':>13}{'match recall':>14}{'entities fully covered':>24}")
for name, mask in [("top-15 overall (current)", r["rank_all"] < 15),
                   ("top-20 overall", r["rank_all"] < 20),
                   ("10 S2 + 10 S3", r["rank_src"] < 10),
                   ("15 S2 + 15 S3", r["rank_src"] < 15),
                   ("20 S2 + 20 S3 (new)", r["rank_src"] < 20),
                   ("50 S2 + 50 S3 (ceiling)", r["rank_src"] < 50)]:
    sub = r[mask]
    found = sub.groupby("source1_entity_id")["hit"].sum().reindex(ntrue.index, fill_value=0)
    ns = ntrue > 0
    print(f"{name:<26}{sub.groupby('source1_entity_id').size().mean():>13.1f}"
          f"{found[ns].sum() / ntrue[ns].sum():>14.4f}{(found[ns] == ntrue[ns]).mean():>24.4f}")
