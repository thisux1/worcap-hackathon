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
from worcap.pixel_ridge import (load_shared, fit_pixel_ridge, eval_year,
                                eval_year_preds, eval_year_rec)
from worcap.folds import loyo_folds
from worcap.features import production_cols

LOOKUP = {f["test_year"]: f["train_years"] for f in loyo_folds()}

LAM1, LAM2 = 1000.0, 300.0
YEARS = [1983, 1988, 1997, 1998, 2005, 2010, 2015]
WS = [0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
GS = [0.6, 0.7, 0.8, 0.9, 1.0]

from worcap.config import EXT_DIR

t0 = time.time()
store = Store()
alvo, _, tgt_month = load_shared()
df = production_cols(pd.read_parquet(EXT_DIR / "features_shared.parquet"))
Xraw = df.reindex(pd.DatetimeIndex(alvo.time.values)).fillna(0.0) \
        .values.astype(np.float64)
Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
print(f"setup {time.time()-t0:.0f}s | X {Xraw.shape}", flush=True)

preds = {1: [], 2: []}
rec = {}
for yr in YEARS:
    m = fit_pixel_ridge(store, alvo, Xraw, tgt_month, LOOKUP[yr],
                        LAM1, LAM2, use_nmme=False)
    r1, r2 = eval_year(store, alvo, m, yr, 1), eval_year(store, alvo, m, yr, 2)
    for g in (1, 2):
        preds[g] += eval_year_preds(store, alvo, m, yr, g)
    rec[yr] = eval_year_rec(store, alvo, m, yr)
    print(f"{yr}: s1/s2 gap1 {r1} gap2 {r2} | rec {rec[yr][0]:.4f} "
          f"| {time.time()-t0:.0f}s", flush=True)


def rmse_of(months, w, g):
    e = n = 0.0
    for m in months:
        p1 = np.clip(m["base"], 0, None)
        p2 = np.clip(m["base"] + m["corr"], 0, None)
        b = w * p1 + (1 - w) * p2
        d = np.clip(m["clim"] + g * (b - m["clim"]), 0, None)
        y = m["y"]
        e += np.nansum((d - y) ** 2)
        n += np.isfinite(y).sum()
    return np.sqrt(e / n)


print("\n=== GRID w(ridge) x gamma — gap1 / gap2 ===")
print("w\\g   " + "   ".join(f"{g:.1f}" for g in GS))
for w in WS:
    row1 = [rmse_of(preds[1], w, g) for g in GS]
    row2 = [rmse_of(preds[2], w, g) for g in GS]
    print(f"{w:.2f} | " + " ".join(f"{v:.4f}" for v in row1)
          + "  ||  " + " ".join(f"{v:.4f}" for v in row2), flush=True)

print("\n=== RECURSAO (R2) vs baseline gap2 ===")
for yr in YEARS:
    rm, steps = rec[yr]
    print(f"{yr}: rec {rm:.4f} | steps " + " ".join(f"{s:.3f}" for s in steps))
print(f"done {time.time()-t0:.0f}s", flush=True)
