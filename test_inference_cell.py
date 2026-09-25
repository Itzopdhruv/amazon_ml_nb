# ==================================================================================================
# TEST-SET INFERENCE  ->  submission/matching_results.tsv  +  submission/candidate_pairs.tsv
#
# Paste this as the LAST cell of amazon_ml_entity_resolution_pipeline_colab_3.ipynb and run it in the
# SAME Colab session that trained the reranker. It reproduces the validated pipeline on the test set:
#   bge-m3 dense top-400 (exact search, same country only)  +  lexical IDF top-200 (same country only)
#   -> Reciprocal Rank Fusion -> top-20 from S2 + top-20 from S3 per entity (=40) = candidate_pairs.tsv
#   -> fine-tuned bge-reranker-v2-m3 -> score >= best threshold (0.90) -> one-to-one resolution
#   -> matching_results.tsv, validated against every rule in the problem statement.
# Every expensive step saves to Drive (test_work/) and resumes if the cell is re-run.
# ==================================================================================================
import os, gc, json, math, time, shutil, hashlib, unicodedata, re, glob
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- settings (fall back to the notebook's own config when it's in memory) ----------------
_g = globals()
DRIVE_DIR = _g.get("DRIVE_DIR", "/content/drive/MyDrive/amazon_ml_challenge")
LOCAL_DIR = _g.get("LOCAL_DIR", "/content/local")
MODE_TAG = _g.get("MODE_TAG", "dev")
MODEL_DIR = _g.get("MODEL_DIR", f"{LOCAL_DIR}/models_{MODE_TAG}")
BIENCODER_BASE = _g.get("BIENCODER_BASE", "BAAI/bge-m3")
MAX_SEQ_LENGTH = _g.get("MAX_SEQ_LENGTH", 96)
ENCODE_BATCH_SIZE = _g.get("ENCODE_BATCH_SIZE", 256)
TOP_K_PER_SOURCE = _g.get("TOP_K_PER_SOURCE", 50)
FINAL_K = _g.get("RERANK_PREFILTER_K", 15)          # used only when CANDS_PER_SOURCE is None
CANDS_PER_SOURCE = 20       # top-20 from S2 + top-20 from S3 per entity (by RRF score) = 40 candidates,
                            # all reranked. Set to None for the validated setting: top-15 overall (S2+S3 mixed).
LEXICAL_MAX_DOC_FREQ_FRAC = _g.get("LEXICAL_MAX_DOC_FREQ_FRAC", 0.02)
RRF_K = _g.get("RRF_K", 60)
USE_HYBRID_RETRIEVAL = _g.get("USE_HYBRID_RETRIEVAL", True)
DENSE_K = TOP_K_PER_SOURCE * 8                                     # same dense depth as Cell 11 (400)
LEX_K = TOP_K_PER_SOURCE * 4                                       # same lexical depth as Cell 11 (200)
THRESHOLD = float(_g["best_t"]) if _g.get("best_t") is not None else 0.90   # from Cell 16
# one-to-one resolution: follow what Cell 17 decided on validation; default ON (S1 is deduplicated,
# so an S2/S3 record can belong to at most one S1 entity)
USE_ONE_TO_ONE = bool(_g["f05_after"] >= _g["f05_before"]) if ("f05_after" in _g and "f05_before" in _g) else True
SCORE_BATCH = 256
SCORE_CHUNK = 200_000        # pairs per saved score file (resume granularity)
EMB_CHUNK = 200_000          # rows per saved embedding file (resume granularity)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = next((d for d in (f"{DRIVE_DIR}/test", f"{DRIVE_DIR}/raw_data/test", f"{DRIVE_DIR}/Test")
                 if os.path.exists(f"{d}/test_source1.tsv")), None)
assert TEST_DIR, (f"test_source1.tsv not found. Upload test_source1.tsv, test_source2.tsv, test_source3.tsv "
                  f"to {DRIVE_DIR}/test/")
WORK_T = f"{DRIVE_DIR}/test_work"
OUT_DIR = f"{DRIVE_DIR}/submission"
for d in (WORK_T, f"{WORK_T}/emb", f"{WORK_T}/scores", OUT_DIR, f"{LOCAL_DIR}/test_emb"):
    os.makedirs(d, exist_ok=True)

def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)

def _save_npy(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)

log(f"test dir: {TEST_DIR} | threshold {THRESHOLD:.2f} | one-to-one {USE_ONE_TO_ONE} | "
    f"hybrid {USE_HYBRID_RETRIEVAL} | device {DEVICE}")

# ---------------- 0. back up the trained reranker to Drive FIRST ----------------
# (the notebook's sync_to_drive("models") looks for a folder literally named "models", but the real
#  folder is models_dev, so the trained reranker was never copied to Drive. Fix that now.)
RR_LOCAL = f"{MODEL_DIR}/reranker-finetuned"
RR_DRIVE = f"{DRIVE_DIR}/reranker_backup_{MODE_TAG}/reranker-finetuned"
if os.path.isdir(RR_LOCAL) and not os.path.exists(f"{RR_DRIVE}/_COPY_DONE"):
    shutil.copytree(RR_LOCAL, RR_DRIVE, dirs_exist_ok=True)
    open(f"{RR_DRIVE}/_COPY_DONE", "w").close()
    log(f"Backed up the trained reranker to Drive: {RR_DRIVE}")

# ---------------- 1. models ----------------
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

if _g.get("reranker") is None:
    rr_path = RR_LOCAL if os.path.isdir(RR_LOCAL) else (RR_DRIVE if os.path.exists(f"{RR_DRIVE}/_COPY_DONE") else None)
    assert rr_path, ("No trained reranker found (not in memory, not in MODEL_DIR, not on Drive). "
                     "Re-run the training cells (13 + 14a) first.")
    reranker = CrossEncoder(rr_path, num_labels=1, max_length=MAX_SEQ_LENGTH * 2)   # same length it was trained with
    log(f"Loaded reranker from {rr_path}")
if DEVICE == "cuda":
    reranker.model.half()
reranker.model.eval()

if _g.get("bi_model") is None:
    bi_model = SentenceTransformer(BIENCODER_BASE)
    bi_model.max_seq_length = MAX_SEQ_LENGTH
    log(f"Loaded {BIENCODER_BASE} (off-the-shelf, as in training)")
if DEVICE == "cuda":
    bi_model.half()      # fp16 inference: ~2x faster, cosine rankings effectively unchanged
bi_model.eval()

# ---------------- 2. load + normalize the test files (identical normalization to training) ----------------
_SUFFIX = {"pvt": "private", "private": "private", "ltd": "limited", "limited": "limited", "llc": "llc",
           "inc": "incorporated", "incorporated": "incorporated", "corp": "corporation",
           "corporation": "corporation", "co": "company", "company": "company", "llp": "llp", "dba": "dba"}
_ADDR = {"rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
         "apt": "apartment", "ste": "suite", "hwy": "highway", "ct": "court", "pl": "place", "sq": "square"}
_ws = re.compile(r"\s+")

def _norm(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("&", " and ")
    s = "".join(ch if (unicodedata.category(ch)[0] in ("L", "M", "N") or ch.isspace()) else " " for ch in s)
    return _ws.sub(" ", s).strip()

def _prep_test(src):
    out = f"{WORK_T}/test_{src}_prep.parquet"
    if os.path.exists(out):
        return pd.read_parquet(out)
    df = pd.read_csv(f"{TEST_DIR}/test_source{src[1]}.tsv", sep="\t", dtype=str, keep_default_na=False).fillna("")
    for c in ("entity_id", "business_name", "business_address", "country"):
        assert c in df.columns, f"test_source{src[1]}.tsv missing column {c}"
    names = [" ".join(_SUFFIX.get(t, t) for t in _norm(x).split()) for x in df["business_name"].values]
    addrs = [" ".join(_ADDR.get(t, t) for t in _norm(x).split()) for x in df["business_address"].values]
    res = pd.DataFrame({
        "entity_id": df["entity_id"].str.strip().values,
        "country": df["country"].str.strip().values,
        "text": [f"name: {n} | address: {a}".strip() for n, a in zip(names, addrs)],   # same as Cell 5
    })
    assert res["entity_id"].is_unique, f"duplicate entity_id in test_source{src[1]}.tsv"
    res.to_parquet(out + ".tmp", index=False); os.replace(out + ".tmp", out)
    return res

t1, t2, t3 = _prep_test("s1"), _prep_test("s2"), _prep_test("s3")
S1_IDS, S1_TEXT, S1_CTRY = t1["entity_id"].values, t1["text"].values, t1["country"].values
C_IDS = np.concatenate([t2["entity_id"].values, t3["entity_id"].values])       # corpus = S2 rows then S3 rows
C_TEXT = np.concatenate([t2["text"].values, t3["text"].values])
C_CTRY = np.concatenate([t2["country"].values, t3["country"].values])
N_S2 = len(t2)
log(f"test: S1={len(t1):,}  S2={len(t2):,}  S3={len(t3):,}")
for c in sorted(set(S1_CTRY)):
    log(f"   {c:>8}: S1={(S1_CTRY == c).sum():,}  S2+S3={(C_CTRY == c).sum():,}")

# ---------------- 3. bge-m3 embeddings (fp16, chunked; cached on Drive + fast local copy) ----------------
def _emb_paths(src, n):
    return [f"{WORK_T}/emb/{src}_{i:04d}.npy" for i in range(math.ceil(n / EMB_CHUNK))]

def embed(src, texts):
    paths = _emb_paths(src, len(texts))
    todo = [i for i, p in enumerate(paths) if not os.path.exists(p)]
    t0, done = time.time(), 0
    for i, p in enumerate(paths):
        local = f"{LOCAL_DIR}/test_emb/{os.path.basename(p)}"
        if os.path.exists(p):
            if not os.path.exists(local):
                shutil.copy(p, local)
            continue
        chunk = list(texts[i * EMB_CHUNK:(i + 1) * EMB_CHUNK])
        with torch.inference_mode():
            E = bi_model.encode(chunk, batch_size=ENCODE_BATCH_SIZE, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True).astype(np.float16)
        _save_npy(local, E)
        shutil.copy(local, p + ".tmp"); os.replace(p + ".tmp", p)
        done += len(chunk)
        left = sum(min(EMB_CHUNK, len(texts) - j * EMB_CHUNK) for j in todo if j > i)
        rate = done / (time.time() - t0)
        log(f"  embedded {src} chunk {i + 1}/{len(paths)} | {rate:,.0f} rows/s | ETA {left / rate / 60:.0f} min")
    return [f"{LOCAL_DIR}/test_emb/{os.path.basename(p)}" for p in paths]

EMB_S1 = embed("s1", S1_TEXT)
EMB_C = embed("s2", t2["text"].values) + embed("s3", t3["text"].values)
# corpus row offset of each embedding chunk (S2 chunks first, then S3 chunks, same order as C_IDS)
C_CHUNK_START = np.cumsum([0] + [np.load(p, mmap_mode="r").shape[0] for p in EMB_C])[:-1]
assert C_CHUNK_START[-1] + np.load(EMB_C[-1], mmap_mode="r").shape[0] == len(C_IDS)
del t2, t3
gc.collect()

# ---------------- 4. candidate generation (exact dense + lexical, RRF, then top-K) ----------------
CAND_TAG = f"s2x{CANDS_PER_SOURCE}_s3x{CANDS_PER_SOURCE}" if CANDS_PER_SOURCE else f"top{FINAL_K}"
CAND_W = 2 * CANDS_PER_SOURCE if CANDS_PER_SOURCE else FINAL_K     # candidate slots per entity
# each candidate setting has its own cache files (top15 keeps the original names from earlier runs)
cand_path = f"{WORK_T}/candidates.npz" if CAND_TAG == "top15" else f"{WORK_T}/candidates_{CAND_TAG}.npz"
score_dir = f"{WORK_T}/scores" if CAND_TAG == "top15" else f"{WORK_T}/scores_{CAND_TAG}"
log(f"candidate setting: {CAND_TAG} ({CAND_W} per entity)")
n_q = len(S1_IDS)
_codes = {c: i for i, c in enumerate(sorted(set(S1_CTRY) | set(C_CTRY)))}
q_cc = np.array([_codes[c] for c in S1_CTRY], np.int32)
c_cc = np.array([_codes[c] for c in C_CTRY], np.int32)
_code_name = {i: c for c, i in _codes.items()}

# ---- 4a. exact dense top-DENSE_K per query, same country only (streams the corpus chunk by chunk) ----
dense_path = f"{WORK_T}/dense_idx.npy"
if os.path.exists(cand_path):
    pass
elif os.path.exists(dense_path):
    DENSE_IDX = np.load(dense_path)
    log("loaded cached dense results")
else:
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    t0 = time.time()
    DENSE_IDX = np.full((n_q, DENSE_K), -1, np.int32)
    group = max(1, int(2.5e9 // (DENSE_K * 12)))             # queries per pass, keeps top-K buffers ~2.5 GB
    Q_ALL = np.concatenate([np.load(p) for p in EMB_S1])
    qb = 2048
    for g0 in range(0, n_q, group):
        g_rows = np.arange(g0, min(n_q, g0 + group))
        Q = torch.from_numpy(Q_ALL[g_rows]).to(DEVICE, dtype)
        by_cc = {int(c): torch.from_numpy(np.nonzero(q_cc[g_rows] == c)[0]).to(DEVICE) for c in np.unique(q_cc[g_rows])}
        best_s = torch.full((len(g_rows), DENSE_K), -float("inf"), device=DEVICE)
        best_i = torch.full((len(g_rows), DENSE_K), -1, dtype=torch.int64, device=DEVICE)
        for p, start in zip(EMB_C, C_CHUNK_START):
            E = np.load(p)
            cc_chunk = c_cc[start:start + len(E)]
            for cc, q_idx in by_cc.items():
                cm = np.nonzero(cc_chunk == cc)[0]
                if len(cm) == 0:
                    continue
                Ec = torch.from_numpy(E[cm]).to(DEVICE, dtype)
                gidx = torch.from_numpy(start + cm).to(DEVICE)
                b = 0
                while b < len(q_idx):
                    qi = q_idx[b:b + qb]
                    try:
                        S = Q[qi] @ Ec.T
                        ts, ti = S.topk(min(DENSE_K, S.shape[1]), dim=1)
                        cs = torch.cat([best_s[qi], ts.float()], 1)
                        ci = torch.cat([best_i[qi], gidx[ti]], 1)
                        ns, pos = cs.topk(DENSE_K, dim=1)
                        best_s[qi], best_i[qi] = ns, ci.gather(1, pos)
                        b += len(qi)
                    except torch.cuda.OutOfMemoryError:
                        S = ts = ti = cs = ci = None
                        torch.cuda.empty_cache()
                        qb = max(1, qb // 2)
                        log(f"  OOM in dense search, query block -> {qb}")
                    S = None
                del Ec, gidx
            del E
        DENSE_IDX[g_rows] = best_i.cpu().numpy().astype(np.int32)
        del Q, best_s, best_i
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        el = time.time() - t0
        log(f"  dense search: {g_rows[-1] + 1:,}/{n_q:,} queries | ETA {el / (g_rows[-1] + 1) * (n_q - g_rows[-1] - 1) / 60:.0f} min")
    del Q_ALL
    _save_npy(dense_path, DENSE_IDX)
    log("dense results saved")

# ---- 4b. lexical IDF retrieval, identical scoring to Cell 10/11: sum of idf*tf over the query's distinct
#          tokens; tokens in more than LEXICAL_MAX_DOC_FREQ_FRAC of the country's corpus are dropped.
#          Runs on all CPU cores; results saved per segment so a disconnect loses at most one segment. ----
import multiprocessing as mp
LEX_QBLOCK = 128           # queries per sparse product (bounds per-worker memory)
LEX_SEGMENT = 100_000      # queries per saved segment (resume granularity)
os.makedirs(f"{WORK_T}/lex", exist_ok=True)
_LEX = {}

def _lex_block(b0):
    rows = _LEX["rows"][b0:b0 + LEX_QBLOCK]
    SC = ((_LEX["qcv"].transform(S1_TEXT[rows]) @ _LEX["idf"]) @ _LEX["DT"]).tocsr()   # queries x docs
    out = np.full((len(rows), LEX_K), -1, np.int32)
    for r in range(SC.shape[0]):
        lo, hi = SC.indptr[r], SC.indptr[r + 1]
        if lo == hi:
            continue
        d, j = SC.data[lo:hi], SC.indices[lo:hi]
        if len(d) > LEX_K:
            top = np.argpartition(-d, LEX_K - 1)[:LEX_K]
            d, j = d[top], j[top]
        o = np.argsort(-d, kind="stable")
        out[r, :len(o)] = _LEX["c_rows"][j[o]]
    return b0, out

if USE_HYBRID_RETRIEVAL and not os.path.exists(cand_path):
    LEX_IDX = np.full((n_q, LEX_K), -1, np.int32)
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    for cc in np.unique(q_cc):
        name = _code_name[int(cc)]
        q_rows = np.nonzero(q_cc == cc)[0]
        c_rows = np.nonzero(c_cc == cc)[0]
        segs = [(s0, f"{WORK_T}/lex/{name}_{s0 // LEX_SEGMENT:04d}.npy") for s0 in range(0, len(q_rows), LEX_SEGMENT)]
        if len(c_rows) == 0 or all(os.path.exists(sp) for _, sp in segs):
            for s0, sp in segs:
                if os.path.exists(sp):
                    LEX_IDX[q_rows[s0:s0 + LEX_SEGMENT]] = np.load(sp)
            continue
        log(f"  lexical index for {name}: {len(c_rows):,} docs ...")
        cv = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, dtype=np.float32)
        D = cv.fit_transform(C_TEXT[c_rows]).tocsr()                  # docs x vocab, term frequencies
        df_ = np.bincount(D.indices, minlength=D.shape[1])             # document frequency per token
        keep = np.nonzero(df_ <= max(1, int(len(c_rows) * LEXICAL_MAX_DOC_FREQ_FRAC)))[0]
        feats = cv.get_feature_names_out()
        _LEX.clear()
        _LEX.update(
            rows=None, c_rows=c_rows.astype(np.int32),
            DT=D[:, keep].T.tocsr(),                                   # kept vocab x docs
            idf=sparse.diags(np.log1p(len(c_rows) / df_[keep]).astype(np.float32)),
            qcv=CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, binary=True,
                                vocabulary={t: i for i, t in enumerate(feats[keep])}, dtype=np.float32),
        )
        del D, cv, feats
        gc.collect()
        log(f"  {name}: kept {len(keep):,} tokens (df <= {LEXICAL_MAX_DOC_FREQ_FRAC:.0%}), {n_workers} workers")
        for s0, sp in segs:
            seg_rows = q_rows[s0:s0 + LEX_SEGMENT]
            if os.path.exists(sp):
                LEX_IDX[seg_rows] = np.load(sp)
                continue
            _LEX["rows"] = seg_rows
            seg_out = np.full((len(seg_rows), LEX_K), -1, np.int32)
            starts = range(0, len(seg_rows), LEX_QBLOCK)
            try:
                with mp.get_context("fork").Pool(n_workers) as pool:
                    for b0, out in pool.imap_unordered(_lex_block, starts, chunksize=4):
                        seg_out[b0:b0 + len(out)] = out
            except Exception as e:                                     # fall back to a single process
                log(f"  parallel lexical search failed ({type(e).__name__}: {e}); running single-process")
                for b0 in starts:
                    _, out = _lex_block(b0)
                    seg_out[b0:b0 + len(out)] = out
            _save_npy(sp, seg_out)
            LEX_IDX[seg_rows] = seg_out
            done = s0 + len(seg_rows)
            log(f"  lexical {name}: {done:,}/{len(q_rows):,} queries ({(time.time() - t0) / 60:.0f} min so far)")
        _LEX.clear()
        gc.collect()

# ---- 4c. Reciprocal Rank Fusion -> top-K per query (per source, or overall), vectorized ----
if os.path.exists(cand_path):
    with np.load(cand_path) as z:
        CAND_IDX, CAND_RRF = z["idx"], z["rrf"]
    log(f"loaded cached candidates {CAND_IDX.shape}")
else:
    CAND_IDX = np.full((n_q, CAND_W), -1, np.int64)
    CAND_RRF = np.zeros((n_q, CAND_W), np.float32)
    N_C = len(C_IDS)
    RB = 50_000
    for b0 in range(0, n_q, RB):
        sl = slice(b0, min(n_q, b0 + RB))
        lists = [DENSE_IDX[sl]] + ([LEX_IDX[sl]] if USE_HYBRID_RETRIEVAL else [])
        qs, cs, ws = [], [], []
        for L in lists:
            r, k = np.nonzero(L >= 0)
            qs.append(r); cs.append(L[r, k].astype(np.int64)); ws.append(1.0 / (RRF_K + k + 1))
        q_, c_, w_ = np.concatenate(qs), np.concatenate(cs), np.concatenate(ws)
        key = q_.astype(np.int64) * N_C + c_
        uk, inv = np.unique(key, return_inverse=True)
        score = np.bincount(inv, weights=w_)
        uq, uc = uk // N_C, uk % N_C
        if CANDS_PER_SOURCE:
            src = (uc >= N_S2).astype(np.int64)                           # 0 = S2, 1 = S3
            order = np.lexsort((-score, src, uq))                         # by query, source, score desc
            uq, uc, score, src = uq[order], uc[order], score[order], src[order]
            grp = uq * 2 + src
            rank = np.arange(len(grp)) - np.searchsorted(grp, grp, side="left")
            m = rank < CANDS_PER_SOURCE
            col = src[m] * CANDS_PER_SOURCE + rank[m]                     # S2 in cols 0..19, S3 in 20..39
        else:
            order = np.lexsort((-score, uq))                              # by query, then score desc
            uq, uc, score = uq[order], uc[order], score[order]
            rank = np.arange(len(uq)) - np.searchsorted(uq, uq, side="left")
            m = rank < FINAL_K
            col = rank[m]
        CAND_IDX[b0 + uq[m], col] = uc[m]
        CAND_RRF[b0 + uq[m], col] = score[m]
    DENSE_IDX = LEX_IDX = None
    gc.collect()
    np.savez(cand_path + ".tmp.npz", idx=CAND_IDX, rrf=CAND_RRF)
    os.replace(cand_path + ".tmp.npz", cand_path)
    log("candidates saved")

# country hard-filter sanity check
_v = CAND_IDX >= 0
assert (C_CTRY[CAND_IDX[_v]] == np.repeat(S1_CTRY, _v.sum(1))).all(), "cross-country candidate found"

def _write(path, col, lists):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for sid, ids in zip(S1_IDS, lists):
            f.write(f"{sid}\t{','.join(ids)}\n")
    os.replace(tmp, path)

cand_lists = [[C_IDS[j] for j in row if j >= 0] for row in CAND_IDX]
CAND_TSV = f"{OUT_DIR}/candidate_pairs.tsv"
_write(CAND_TSV, "candidate_entity_ids", cand_lists)
n_c = np.array([len(x) for x in cand_lists])
log(f"candidate_pairs.tsv written: avg {n_c.mean():.1f} candidates/entity, {(n_c == 0).sum():,} entities with none")
if CANDS_PER_SOURCE:
    _n2 = ((CAND_IDX >= 0) & (CAND_IDX < N_S2)).sum(1)
    log(f"   per entity: avg {_n2.mean():.1f} from S2, {(n_c - _n2).mean():.1f} from S3")

# ---------------- 5. cross-encoder scoring (chunked, resumable) ----------------
qi, kj = np.nonzero(CAND_IDX >= 0)
pair_c = CAND_IDX[qi, kj]
sig = hashlib.md5(qi.tobytes() + pair_c.tobytes()).hexdigest()
os.makedirs(score_dir, exist_ok=True)
meta_p = f"{score_dir}/meta.json"
if os.path.exists(meta_p) and json.load(open(meta_p)).get("sig") != sig:
    log("candidates changed since last scoring run: clearing old scores")
    shutil.rmtree(score_dir); os.makedirs(score_dir)
json.dump({"sig": sig, "n": int(len(qi))}, open(meta_p, "w"))
N_C = len(C_IDS)

def _known_scores():
    """(sorted pair keys, scores) already computed under ANY other candidate setting, so they are
    looked up instead of re-scored (top-15 overall is a subset of top-20 per source)."""
    keys, vals = [], []
    others = [(f"{WORK_T}/candidates.npz", f"{WORK_T}/scores")] + [
        (cp, f"{WORK_T}/scores_{os.path.basename(cp)[len('candidates_'):-4]}")
        for cp in glob.glob(f"{WORK_T}/candidates_*.npz")]
    for cp, sd in others:
        if sd == score_dir or not (os.path.exists(cp) and os.path.exists(f"{sd}/meta.json")):
            continue
        with np.load(cp) as z:
            ci = z["idx"]
        oq, ok_ = np.nonzero(ci >= 0)
        oc = ci[oq, ok_]
        if json.load(open(f"{sd}/meta.json")).get("sig") != hashlib.md5(oq.tobytes() + oc.tobytes()).hexdigest():
            continue
        okey = oq.astype(np.int64) * N_C + oc
        for f in glob.glob(f"{sd}/chunk_*.npy"):
            c0 = int(os.path.basename(f)[6:11]) * SCORE_CHUNK
            v = np.load(f)
            keys.append(okey[c0:c0 + len(v)]); vals.append(v)
    if not keys:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    k, v = np.concatenate(keys), np.concatenate(vals)
    k, first = np.unique(k, return_index=True)
    return k, v[first]

n_chunks = math.ceil(len(qi) / SCORE_CHUNK)
todo_chunks = [c for c in range(n_chunks) if not os.path.exists(f"{score_dir}/chunk_{c:05d}.npy")]
KNOWN_K, KNOWN_V = _known_scores() if todo_chunks else (np.zeros(0, np.int64), np.zeros(0, np.float32))
if len(KNOWN_K):
    log(f"reusing {len(KNOWN_K):,} pair scores from earlier runs (no re-scoring needed for those)")
pair_key = qi.astype(np.int64) * N_C + pair_c
t0, done, reused, bs = time.time(), 0, 0, SCORE_BATCH
for c in todo_chunks:
    p = f"{score_dir}/chunk_{c:05d}.npy"
    sl = slice(c * SCORE_CHUNK, (c + 1) * SCORE_CHUNK)
    s = np.full(len(pair_key[sl]), np.nan, np.float32)
    if len(KNOWN_K):
        pos = np.minimum(np.searchsorted(KNOWN_K, pair_key[sl]), len(KNOWN_K) - 1)
        hit = KNOWN_K[pos] == pair_key[sl]
        s[hit] = KNOWN_V[pos[hit]]
        reused += int(hit.sum())
    need = np.nonzero(np.isnan(s))[0]
    if len(need):
        a, b = S1_TEXT[qi[sl]][need], C_TEXT[pair_c[sl]][need]
        order = np.argsort(np.fromiter((len(x) + len(y) for x, y in zip(a, b)), np.int64, len(a)), kind="stable")
        pairs = [(a[i], b[i]) for i in order]
        while True:
            try:
                s_sorted = reranker.predict(pairs, batch_size=bs, activation_fn=torch.nn.Sigmoid(),
                                            show_progress_bar=False, convert_to_numpy=True)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                log(f"  OOM while scoring, batch -> {bs}")
        tmp = np.empty(len(order), np.float32)
        tmp[order] = np.asarray(s_sorted, dtype=np.float32).reshape(-1)
        s[need] = tmp
        done += len(need)
    assert not np.isnan(s).any()
    _save_npy(p, s)
    rate = done / max(time.time() - t0, 1e-9)
    left = sum(min(SCORE_CHUNK, len(qi) - j * SCORE_CHUNK) for j in todo_chunks if j > c)
    eta = f"ETA {left / rate / 60:.0f} min" if done else "ETA -"
    log(f"  scored chunk {c + 1}/{n_chunks} | {rate:,.0f} new pairs/s | reused {reused:,} so far | {eta}")
SCORES = np.concatenate([np.load(f"{score_dir}/chunk_{c:05d}.npy") for c in range(n_chunks)]) if n_chunks else np.zeros(0, np.float32)
json.dump({"cand_path": cand_path, "score_dir": score_dir, "tag": CAND_TAG}, open(f"{WORK_T}/current.json", "w"))

# ---------------- 6. threshold + one-to-one resolution -> matching_results.tsv ----------------
keep = SCORES >= THRESHOLD
pred = pd.DataFrame({"q": qi[keep], "c": pair_c[keep], "s": SCORES[keep]})
n_before = len(pred)
if USE_ONE_TO_ONE and len(pred):
    pred = pred.loc[pred.groupby("c")["s"].idxmax()]          # each S2/S3 record -> its best S1 entity only
log(f"predicted links: {n_before:,} above threshold, {len(pred):,} after one-to-one resolution")

match_lists = [[] for _ in range(len(S1_IDS))]
pred = pred.sort_values(["q", "s"], ascending=[True, False])
for q_, c_ in zip(pred["q"].values, pred["c"].values):
    match_lists[q_].append(C_IDS[c_])
MATCH_TSV = f"{OUT_DIR}/matching_results.tsv"
_write(MATCH_TSV, "matched_entity_ids", match_lists)

n_m = np.array([len(x) for x in match_lists])
log(f"matching_results.tsv: {len(n_m):,} rows | avg {n_m.mean():.2f} matches | "
    f"empty (predicted singletons) {(n_m == 0).mean():.2%}  (train ground truth: ~5.6%)")
for c in sorted(set(S1_CTRY)):
    m = S1_CTRY == c
    log(f"   {c:>8}: avg {n_m[m].mean():.2f} matches, {(n_m[m] == 0).mean():.2%} empty")

# ---------------- 7. validate against the problem statement's rules ----------------
problems = []
valid = set(C_IDS)
tabs = {}
for path, col in ((MATCH_TSV, "matched_entity_ids"), (CAND_TSV, "candidate_entity_ids")):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(df.columns) != ["source1_entity_id", col]:
        problems.append(f"{path}: header {list(df.columns)}")
    if df["source1_entity_id"].duplicated().any():
        problems.append(f"{path}: duplicate source1_entity_id rows")
    if len(df) != len(S1_IDS) or set(df["source1_entity_id"]) != set(S1_IDS):
        problems.append(f"{path}: rows don't match test_source1 exactly")
    lists = {}
    for sid, v in zip(df["source1_entity_id"], df[col]):
        ids = v.split(",") if v else []
        if len(ids) != len(set(ids)):
            problems.append(f"{path}: duplicate ids for {sid}")
        if any((x not in valid) or not x.startswith(("S2-", "S3-")) for x in ids):
            problems.append(f"{path}: invalid id for {sid}")
        lists[sid] = set(ids)
    tabs[col] = lists
if any(not m <= tabs["candidate_entity_ids"][s] for s, m in tabs["matched_entity_ids"].items()):
    problems.append("some matches are not in candidate_pairs")
if problems:
    print("VALIDATION PROBLEMS:"); [print("  -", p) for p in problems[:20]]
else:
    print(f"\nPASS: both files follow every rule.\n  upload this to the portal: {MATCH_TSV}\n  candidates: {CAND_TSV}")
    try:
        from google.colab import files as _colab_files
        _colab_files.download(MATCH_TSV)     # also triggers a browser download
    except Exception:
        pass
