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
from worcap.pixel_ridge import load_shared, fit_pixel_ridge, eval_year
from worcap.folds import loyo_folds
from worcap.features import production_cols

LOOKUP = {f["test_year"]: f["train_years"] for f in loyo_folds()}

YEARS = [1988, 1997, 2005, 2015]
LAM1S = [100.0, 300.0, 1000.0, 3000.0, 10000.0]
LAM2S = [30.0, 100.0, 300.0, 1000.0]

from worcap.config import EXT_DIR

t0 = time.time()
store = Store()
alvo, _, tgt_month = load_shared()
df = production_cols(pd.read_parquet(EXT_DIR / "features_shared.parquet"))
Xraw = df.reindex(pd.DatetimeIndex(alvo.time.values)).fillna(0.0) \
        .values.astype(np.float64)
Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
print(f"setup {time.time()-t0:.0f}s | X {Xraw.shape}", flush=True)

rows = {}
for yr in YEARS:
    for l1 in LAM1S:
        for l2 in LAM2S:
            m = fit_pixel_ridge(store, alvo, Xraw, tgt_month, LOOKUP[yr],
                                l1, l2, use_nmme=False)
            r1, r2 = eval_year(store, alvo, m, yr, 1)
            q1, q2 = eval_year(store, alvo, m, yr, 2)
            rows[(yr, l1, l2)] = (r1[0], r1[1], q1[0], q2[1])
            print(f"{yr} l1={l1:g} l2={l2:g} | gap1 s1 {r1[0]:.4f} "
                  f"s12 {r1[1]:.4f} | gap2 s1 {q1[0]:.4f} s12 {q2[1]:.4f} "
                  f"| {time.time()-t0:.0f}s", flush=True)

print("\n=== MEDIA gap1 (s1 / s1+2) ===")
print("l1\\l2   " + "      ".join(f"{l2:g}" for l2 in LAM2S))
for l1 in LAM1S:
    r1m = [np.mean([rows[(y, l1, l2)][0] for y in YEARS]) for l2 in LAM2S]
    r2m = [np.mean([rows[(y, l1, l2)][1] for y in YEARS]) for l2 in LAM2S]
    print(f"{l1:g} | " + " ".join(f"{v:.4f}" for v in r1m)
          + "  ||  " + " ".join(f"{v:.4f}" for v in r2m), flush=True)

print("\n=== MEDIA gap2 (s1 / s1+2) ===")
print("l1\\l2   " + "      ".join(f"{l2:g}" for l2 in LAM2S))
for l1 in LAM1S:
    r1m = [np.mean([rows[(y, l1, l2)][2] for y in YEARS]) for l2 in LAM2S]
    r2m = [np.mean([rows[(y, l1, l2)][3] for y in YEARS]) for l2 in LAM2S]
    print(f"{l1:g} | " + " ".join(f"{v:.4f}" for v in r1m)
          + "  ||  " + " ".join(f"{v:.4f}" for v in r2m), flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
