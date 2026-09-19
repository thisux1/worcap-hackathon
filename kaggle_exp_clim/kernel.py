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

import worcap.pixel_ridge as pr
import worcap.climatology as cl
assert hasattr(pr, "eval_year_rec"), "dataset worcap-assets2 desatualizado"
from worcap.gbm_data import Store
from worcap.pixel_ridge import load_shared, eval_year
from worcap.folds import loyo_folds
from worcap.features import production_cols

LOOKUP = {f["test_year"]: f["train_years"] for f in loyo_folds()}

LAM1, LAM2 = 1000.0, 300.0
YEARS = [1988, 1997, 2005, 2010, 2015]

ORIG = cl.fit_climatology

def make_fit(mode):
    def f(tp, years, trend=False):
        ys = sorted(years)
        if mode == "full":
            return ORIG(tp, ys)
        if mode == "trend":
            return ORIG(tp, ys, trend=True)
        win = int(mode)
        return ORIG(tp, ys[-win:])
    return f

from worcap.config import EXT_DIR

t0 = time.time()
store = Store()
alvo, _, tgt_month = load_shared()
df = production_cols(pd.read_parquet(EXT_DIR / "features_shared.parquet"))
Xraw = df.reindex(pd.DatetimeIndex(alvo.time.values)).fillna(0.0) \
        .values.astype(np.float64)
Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
print(f"setup {time.time()-t0:.0f}s | X {Xraw.shape}", flush=True)

res = {}
for mode in ["full", "30", "20", "trend"]:
    pr.fit_climatology = make_fit(mode)
    for yr in YEARS:
        m = pr.fit_pixel_ridge(store, alvo, Xraw, tgt_month, LOOKUP[yr],
                               LAM1, LAM2, use_nmme=False)
        r1 = eval_year(store, alvo, m, yr, 1)
        r2 = eval_year(store, alvo, m, yr, 2)
        res[(mode, yr)] = (r1[0], r1[1], r2[0], r2[1])
        print(f"{mode:>5} {yr}: gap1 s1 {r1[0]:.4f} s12 {r1[1]:.4f} | "
              f"gap2 s1 {r2[0]:.4f} s12 {r2[1]:.4f} | {time.time()-t0:.0f}s",
              flush=True)

print("\n=== MEDIA por modo (gap1 s1/s12 | gap2 s1/s12) ===")
for mode in ["full", "30", "20", "trend"]:
    a = np.mean([res[(mode, y)][0] for y in YEARS])
    b = np.mean([res[(mode, y)][1] for y in YEARS])
    c = np.mean([res[(mode, y)][2] for y in YEARS])
    d = np.mean([res[(mode, y)][3] for y in YEARS])
    print(f"{mode:>5}: {a:.4f}/{b:.4f} | {c:.4f}/{d:.4f}")
print(f"done {time.time()-t0:.0f}s", flush=True)
