# ==================================================================================================
# THRESHOLD TUNING + MULTIPLE SUBMISSION FILES
#
# Run this AFTER test_inference_cell.py has finished (same session, or a new one: it reloads the
# cached test scores from Drive). It never re-runs the reranker, so it takes seconds to minutes.
#
# Part A: fine threshold search on your validation set (needs the notebook's val_prefiltered, gt_map,
#         val_ids_use in memory; skipped otherwise). Tries two decision rules:
#           single : predict every candidate with score >= t_all
#           top1   : same, but if nothing passes, still predict the entity's best candidate when its
#                    score >= t_top  (an entity with real matches and an empty prediction scores 0)
# Part B: writes one validated matching_results.tsv per setting into submission/variants/<name>/,
#         so you can upload the best one first and try the others on the leaderboard.
# ==================================================================================================
import os, json, glob
import numpy as np
import pandas as pd

_g = globals()
DRIVE_DIR = _g.get("DRIVE_DIR", "/content/drive/MyDrive/amazon_ml_challenge")
WORK_T = f"{DRIVE_DIR}/test_work"
VAR_DIR = f"{DRIVE_DIR}/submission/variants"
os.makedirs(VAR_DIR, exist_ok=True)
USE_ONE_TO_ONE = _g.get("USE_ONE_TO_ONE", True)
EXTRA_THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.97]     # also exported, single rule, for leaderboard checks

def f05_counts(tp, npred, ntrue):
    tp, npred, ntrue = (np.asarray(x, np.float64) for x in (tp, npred, ntrue))
    p = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    r = np.divide(tp, ntrue, out=np.zeros_like(tp), where=ntrue > 0)
    den = 0.25 * p + r
    f = np.divide(1.25 * p * r, den, out=np.zeros_like(tp), where=den > 0)
    return np.where(ntrue == 0, (npred == 0).astype(np.float64), f)

assert abs(f05_counts([2], [3], [2])[0] - 0.7142857142857143) < 1e-9     # problem-statement example

def decide(q, c, s, n_ent, t_all, t_top, one_to_one):
    """q, c, s: entity index, candidate id/index, score per pair. Returns boolean keep-mask over pairs."""
    keep = s >= t_all
    if t_top is not None:
        order = np.lexsort((-s, q))
        first = order[np.r_[True, q[order][1:] != q[order][:-1]]]      # best pair of each entity
        has = np.bincount(q[keep], minlength=n_ent) > 0
        add = first[(~has[q[first]]) & (s[first] >= t_top)]
        keep[add] = True
    if one_to_one and keep.any():
        idx = np.nonzero(keep)[0]
        df = pd.DataFrame({"i": idx, "c": c[idx], "s": s[idx]})
        winners = df.loc[df.groupby("c")["s"].idxmax(), "i"].values
        keep = np.zeros_like(keep)
        keep[winners] = True
    return keep

# ---------------- Part A: validation search ----------------
best = {"rule": "single", "t_all": float(_g.get("best_t") or 0.90), "t_top": None, "f05": None}
if all(k in _g for k in ("val_prefiltered", "gt_map", "val_ids_use")) and "reranker_score" in _g["val_prefiltered"]:
    vp = _g["val_prefiltered"]
    ent = {e: i for i, e in enumerate(_g["val_ids_use"])}
    vq = vp["source1_entity_id"].map(ent).values.astype(np.int64)
    vc = vp["candidate_entity_id"].values
    vs = vp["reranker_score"].values.astype(np.float64)
    ok = vq >= 0
    vq, vc, vs = vq[ok], vc[ok], vs[ok]
    gm = _g["gt_map"]
    vlab = np.array([c in gm.get(e, ()) for e, c in zip(vp["source1_entity_id"].values[ok], vc)])
    ntrue = np.array([len(gm.get(e, ())) for e in _g["val_ids_use"]])
    n_ent = len(ntrue)
    _, vc_codes = np.unique(vc, return_inverse=True)

    def val_f05(t_all, t_top, one_to_one=False):
        k = decide(vq, vc_codes, vs, n_ent, t_all, t_top, one_to_one)
        return f05_counts(np.bincount(vq[k & vlab], minlength=n_ent), np.bincount(vq[k], minlength=n_ent), ntrue)

    grid = np.round(np.arange(0.30, 0.996, 0.01), 2)
    rows = []
    for t in grid:
        rows.append(("single", float(t), None, val_f05(t, None).mean()))
        for tt in np.round(np.arange(0.05, t + 1e-9, 0.05), 2):
            rows.append(("top1", float(t), float(tt), val_f05(t, tt).mean()))
    res = pd.DataFrame(rows, columns=["rule", "t_all", "t_top", "f05"]).sort_values("f05", ascending=False)
    print(f"validation: {n_ent:,} entities ({int((ntrue == 0).sum()):,} singletons), {len(vs):,} scored pairs")
    print("\nTop 10 settings (before one-to-one):")
    print(res.head(10).to_string(index=False))
    b = res.iloc[0]
    best = {"rule": b.rule, "t_all": float(b.t_all), "t_top": None if pd.isna(b.t_top) else float(b.t_top)}
    per = val_f05(best["t_all"], best["t_top"], USE_ONE_TO_ONE)
    best["f05"] = float(per.mean())
    single_best = res[res.rule == "single"].iloc[0]
    print(f"\nbest single threshold : t_all={single_best.t_all:.2f}  F0.5={single_best.f05:.4f}")
    print(f"chosen setting        : {best['rule']} t_all={best['t_all']:.2f} t_top={best['t_top']}  "
          f"F0.5={best['f05']:.4f} (with one-to-one={USE_ONE_TO_ONE})")
    print(f"   singletons {per[ntrue == 0].mean():.4f} | non-singletons {per[ntrue > 0].mean():.4f}")
    json.dump(best, open(f"{WORK_T}/best_threshold.json", "w"), indent=2)
else:
    if os.path.exists(f"{WORK_T}/best_threshold.json"):
        best = json.load(open(f"{WORK_T}/best_threshold.json"))
    print(f"validation data not in memory: using {best}")

# ---------------- Part B: load the cached test candidates + scores ----------------
if all(k in _g for k in ("qi", "pair_c", "SCORES", "S1_IDS", "C_IDS")) and len(_g["SCORES"]) == len(_g["qi"]):
    tq, tc, ts, T_S1, T_C = _g["qi"], _g["pair_c"], _g["SCORES"].astype(np.float64), _g["S1_IDS"], _g["C_IDS"]
else:
    cur = json.load(open(f"{WORK_T}/current.json")) if os.path.exists(f"{WORK_T}/current.json") else \
        {"cand_path": f"{WORK_T}/candidates.npz", "score_dir": f"{WORK_T}/scores"}
    print("using", cur)
    with np.load(cur["cand_path"]) as z:
        cand = z["idx"]
    tq, kj = np.nonzero(cand >= 0)
    tc = cand[tq, kj]
    n_chunks = len(glob.glob(f"{cur['score_dir']}/chunk_*.npy"))
    ts = np.concatenate([np.load(f"{cur['score_dir']}/chunk_{i:05d}.npy") for i in range(n_chunks)]).astype(np.float64)
    assert len(ts) == len(tq), "test scoring isn't finished yet: run test_inference_cell.py to the end first"
    T_S1 = pd.read_parquet(f"{WORK_T}/test_s1_prep.parquet", columns=["entity_id"])["entity_id"].values
    T_C = np.concatenate([pd.read_parquet(f"{WORK_T}/test_s{k}_prep.parquet", columns=["entity_id"])["entity_id"].values
                          for k in (2, 3)])
n_test = len(T_S1)
cand_sets = None

def write_variant(name, t_all, t_top):
    global cand_sets
    k = decide(tq, tc, ts, n_test, t_all, t_top, USE_ONE_TO_ONE)
    lists = [[] for _ in range(n_test)]
    order = np.lexsort((-ts[k], tq[k]))
    for q_, c_ in zip(tq[k][order], tc[k][order]):
        lists[q_].append(T_C[c_])
    d = f"{VAR_DIR}/{name}"
    os.makedirs(d, exist_ok=True)
    path = f"{d}/matching_results.tsv"
    with open(path + ".tmp", "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, ids in zip(T_S1, lists):
            f.write(f"{sid}\t{','.join(ids)}\n")
    os.replace(path + ".tmp", path)
    # validation: one row per S1, unique ids, S2/S3 only, subset of candidates
    if cand_sets is None:
        cand_sets = [set() for _ in range(n_test)]
        for q_, c_ in zip(tq, tc):
            cand_sets[q_].add(c_)
    back = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert list(back.columns) == ["source1_entity_id", "matched_entity_ids"] and len(back) == n_test
    assert (back["source1_entity_id"].values == T_S1).all()
    for q_ in np.unique(tq[k]):
        assert len(lists[q_]) == len(set(lists[q_]))
        assert all(x.startswith(("S2-", "S3-")) for x in lists[q_])
    assert all(c_ in cand_sets[q_] for q_, c_ in zip(tq[k], tc[k]))
    n_m = np.array([len(x) for x in lists])
    print(f"  {name:<28} avg {n_m.mean():.2f} matches | empty {(n_m == 0).mean():6.2%} | {path}")

print(f"\nWriting submission variants ({n_test:,} test entities, {len(ts):,} scored pairs, one-to-one={USE_ONE_TO_ONE}):")
tag = f"BEST_{best['rule']}_t{best['t_all']:.2f}" + (f"_top{best['t_top']:.2f}" if best["t_top"] is not None else "")
write_variant(tag, best["t_all"], best["t_top"])
for t in EXTRA_THRESHOLDS:
    write_variant(f"single_t{t:.2f}", t, None)
    write_variant(f"top1_t{t:.2f}_top0.50", t, 0.50)
print(f"\nUpload first: {VAR_DIR}/{tag}/matching_results.tsv")
