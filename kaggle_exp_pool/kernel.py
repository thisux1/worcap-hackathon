import os, sys, time
import numpy as np, pandas as pd, xarray as xr

COMP = "/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul"
ASSETS = "/kaggle/input/worcap-assets2"
if not os.path.exists(ASSETS):
    ASSETS = "/kaggle/input/datasets/thisux1/worcap-assets2"
os.environ.update({
    "WORCAP_DATA": COMP,
    "WORCAP_EXT": ASSETS,
    "WORCAP_SUB": "/kaggle/working",
    "WORCAP_OOF": "/kaggle/working/oof",
    "WORCAP_CACHE": "/tmp",
    "WORCAP_EAGER": "1",
})
sys.path.insert(0, ASSETS)

import worcap.pixel_ridge as _pr
assert hasattr(_pr, "pool_w2"), "dataset worcap-assets2 desatualizado"
from worcap.gbm_data import Store
from worcap.pixel_ridge import (load_shared, fit_pixel_ridge, eval_year,
                                eval_year_preds, pool_w2)
from worcap.folds import loyo_folds
from worcap.features import production_cols

LOOKUP = {f["test_year"]: f["train_years"] for f in loyo_folds()}

LAM1, LAM2 = 1000.0, 300.0
YEARS = [1983, 1988, 1997, 1998, 2005, 2010, 2015]
ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]   # 0 = full smooth, 1 = raw W2
WS = [0.5, 0.65, 0.8]                   # blend: w*s1 + (1-w)*s12

from worcap.config import EXT_DIR

t0 = time.time()
store = Store()
alvo, _, tgt_month = load_shared()
df = production_cols(pd.read_parquet(EXT_DIR / "features_shared.parquet"))
Xraw = df.reindex(pd.DatetimeIndex(alvo.time.values)).fillna(0.0) \
        .values.astype(np.float64)
Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
print(f"setup {time.time()-t0:.0f}s | X {Xraw.shape}", flush=True)

s12 = {(a, g): [] for a in ALPHAS for g in (1, 2)}          # rmse por ano
s1_g = {g: [] for g in (1, 2)}                            # indep. de alpha
sse = {(a, g, w): [0.0, 0.0] for a in ALPHAS for g in (1, 2) for w in WS}
rel = {a: [] for a in ALPHAS}                             # ||W2'-W2||/||W2||
diag = {"a.5_no_intercept": {g: [] for g in (1, 2)},
        "regional.5": {g: [] for g in (1, 2)}}

for yr in YEARS:
    m0 = fit_pixel_ridge(store, alvo, Xraw, tgt_month, LOOKUP[yr],
                         LAM1, LAM2, use_nmme=False)
    W2r = m0["W2"]
    if yr == YEARS[0]:
        # sanity: pooling dentro do fit == pool_w2 pos-hoc sobre W2 cru
        mc = fit_pixel_ridge(store, alvo, Xraw, tgt_month, LOOKUP[yr],
                             LAM1, LAM2, use_nmme=False, pool_alpha=0.5)
        assert np.array_equal(mc["W2_raw"], W2r)
        assert np.allclose(mc["W2"], pool_w2(store, W2r, 0.5))
        print("[ok] fit(pool_alpha=.5) == pool_w2 pos-hoc; W2_raw == W2",
              flush=True)
    for a in ALPHAS:
        m = dict(m0)
        m["W2"] = pool_w2(store, W2r, a) if a < 1 else W2r
        if a < 1:
            rel[a].append(float(np.linalg.norm(m["W2"] - W2r)
                                / np.linalg.norm(W2r)))
        for g in (1, 2):
            r1, r2 = eval_year(store, alvo, m, yr, g)
            s12[(a, g)].append(r2)
            if a == 1.0:
                s1_g[g].append(r1)
            for mo in eval_year_preds(store, alvo, m, yr, g):
                p1 = np.clip(mo["base"], 0, None)
                p2 = np.clip(mo["base"] + mo["corr"], 0, None)
                y = mo["y"]
                n = np.isfinite(y).sum()
                for w in WS:
                    b = np.clip(w * p1 + (1 - w) * p2, 0, None)
                    sse[(a, g, w)][0] += np.nansum((b - y) ** 2)
                    sse[(a, g, w)][1] += n
        print(f"{yr} a={a:.1f} | gap1 s12 {s12[(a,1)][-1]:.4f} "
              f"gap2 s12 {s12[(a,2)][-1]:.4f} | {time.time()-t0:.0f}s",
              flush=True)
    # diagnosticos: intercepto fora do pooling / shrink regional puro
    for tag, kw in [("a.5_no_intercept", dict(alpha=0.5, intercept=False)),
                    ("regional.5", dict(regional=0.5))]:
        md = dict(m0)
        md["W2"] = pool_w2(store, W2r, **kw)
        for g in (1, 2):
            diag[tag][g].append(eval_year(store, alvo, md, yr, g)[1])
    print(f"{yr} diag | a.5-noInt gap1 {diag['a.5_no_intercept'][1][-1]:.4f} "
          f"reg.5 gap1 {diag['regional.5'][1][-1]:.4f}", flush=True)


def rmse(a, g, w):
    e, n = sse[(a, g, w)]
    return np.sqrt(e / n)


print("\n=== STAGE-2 RMSE medio (7 anos) por alpha ===")
print(f"{'alpha':>6} {'gap1':>8} {'gap2':>8} {'mean':>8} {'dW2_rel':>8}")
for a in ALPHAS:
    g1, g2 = np.mean(s12[(a, 1)]), np.mean(s12[(a, 2)])
    print(f"{a:>6.1f} {g1:>8.4f} {g2:>8.4f} {(g1+g2)/2:>8.4f} "
          f"{np.mean(rel[a]) if rel[a] else 0.0:>8.3f}")
print(f"s1 baseline: gap1 {np.mean(s1_g[1]):.4f} gap2 {np.mean(s1_g[2]):.4f}")

print("\n=== S12 por ano (gap1 || gap2) ===")
hdr = "alpha " + " ".join(f"{y}" for y in YEARS)
print(hdr)
for a in ALPHAS:
    print(f"{a:.1f} g1 " + " ".join(f"{v:.3f}" for v in s12[(a, 1)]))
    print(f"{a:.1f} g2 " + " ".join(f"{v:.3f}" for v in s12[(a, 2)]))

print("\n=== BLEND RMSE: w*s1+(1-w)*s12 — (alpha x w) gap1 || gap2 ===")
print("a\\w   " + "   ".join(f"{w:.2f}" for w in WS) + "  ||  "
      + "   ".join(f"{w:.2f}" for w in WS))
for a in ALPHAS:
    r1 = [rmse(a, 1, w) for w in WS]
    r2 = [rmse(a, 2, w) for w in WS]
    print(f"{a:.1f} | " + " ".join(f"{v:.4f}" for v in r1)
          + "  ||  " + " ".join(f"{v:.4f}" for v in r2))

best = min(((rmse(a, 1, w) + rmse(a, 2, w)) / 2, a, w)
           for a in ALPHAS for w in WS)
print(f"\nbest blend: alpha={best[1]} w={best[2]} mean={best[0]:.4f}")
for tag in diag:
    print(f"diag {tag}: gap1 {np.mean(diag[tag][1]):.4f} "
          f"gap2 {np.mean(diag[tag][2]):.4f}")
print(f"done {time.time()-t0:.0f}s", flush=True)
