import os, sys, time
import numpy as np, pandas as pd, xarray as xr

COMP = "/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul"
ASSETS = "/kaggle/input/worcap-assets2"
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
assert hasattr(_pr, "eval_year_rec"), "dataset worcap-assets2 desatualizado"
from worcap.gbm_data import Store
from worcap.pixel_ridge import (load_shared, fit_pixel_ridge,
                                eval_year_preds)
from worcap.folds import loyo_folds
from worcap.features import production_cols

LOOKUP = {f["test_year"]: f["train_years"] for f in loyo_folds()}

LAM1, LAM2 = 1000.0, 300.0
YEARS = [1983, 1988, 1997, 1998, 2005, 2010, 2015]
WS = [0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]

from worcap.config import EXT_DIR

t0 = time.time()
store = Store()
alvo, _, tgt_month = load_shared()
df = production_cols(pd.read_parquet(EXT_DIR / "features_shared.parquet"))
Xraw = df.reindex(pd.DatetimeIndex(alvo.time.values)).fillna(0.0) \
        .values.astype(np.float64)
Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
print(f"setup {time.time()-t0:.0f}s | X {Xraw.shape}", flush=True)

# sse[lag][key] acumulado; lag = pos+1 (gap1) ou pos+13 (gap2)
sse = {l: {"n": 0.0, "s1": 0.0, "s2": 0.0} for l in range(1, 25)}
for w in WS:
    for l in range(1, 25):
        sse[l][w] = 0.0
per_year = {}

for yr in YEARS:
    m = fit_pixel_ridge(store, alvo, Xraw, tgt_month, LOOKUP[yr],
                        LAM1, LAM2, use_nmme=False)
    for g in (1, 2):
        months = eval_year_preds(store, alvo, m, yr, g)
        for pos, mo in enumerate(months):
            lag = pos + 1 + (12 if g == 2 else 0)
            if lag > 24:
                continue
            p1 = np.clip(mo["base"], 0, None)
            p2 = np.clip(mo["base"] + mo["corr"], 0, None)
            y = mo["y"]
            nn = np.isfinite(y).sum()
            sse[lag]["n"] += nn
            sse[lag]["s1"] += np.nansum((p1 - y) ** 2)
            sse[lag]["s2"] += np.nansum((p2 - y) ** 2)
            for w in WS:
                b = np.clip(w * p1 + (1 - w) * p2, 0, None)
                sse[lag][w] += np.nansum((b - y) ** 2)
    print(f"{yr} ok | {time.time()-t0:.0f}s", flush=True)

print("\n=== RMSE por lag: s1 | s2 | melhor w (RMSE) ===")
for l in range(1, 25):
    n = sse[l]["n"]
    if n == 0:
        continue
    r1 = np.sqrt(sse[l]["s1"] / n)
    r2 = np.sqrt(sse[l]["s2"] / n)
    bw = min(WS, key=lambda w: sse[l][w])
    br = np.sqrt(sse[l][bw] / n)
    line = " ".join(f"{w}:{np.sqrt(sse[l][w]/n):.3f}" for w in WS)
    print(f"lag {l:2d} | s1 {r1:.4f} s2 {r2:.4f} | best w={bw} ({br:.4f}) | {line}")

for lo, hi in [(1, 6), (7, 12), (13, 18), (19, 24)]:
    n = sum(sse[l]["n"] for l in range(lo, hi + 1))
    r1 = np.sqrt(sum(sse[l]["s1"] for l in range(lo, hi + 1)) / n)
    r2 = np.sqrt(sum(sse[l]["s2"] for l in range(lo, hi + 1)) / n)
    bw = min(WS, key=lambda w: sum(sse[l][w] for l in range(lo, hi + 1)))
    br = np.sqrt(sum(sse[l][bw] for l in range(lo, hi + 1)) / n)
    print(f"lag {lo}-{hi} | s1 {r1:.4f} s2 {r2:.4f} | best w={bw} ({br:.4f})")
print(f"done {time.time()-t0:.0f}s", flush=True)
