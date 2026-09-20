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
import worcap.enso as _enso
assert hasattr(_pr, "eval_year_preds"), "dataset worcap-assets2 desatualizado"
from worcap.gbm_data import Store
from worcap.pixel_ridge import load_shared, fit_pixel_ridge, eval_year_preds
from worcap.folds import loyo_folds
from worcap.features import production_cols
from worcap.enso import phase_of, phase_composites, enso_offset

LOOKUP = {f["test_year"]: f["train_years"] for f in loyo_folds()}

LAM1, LAM2 = 1000.0, 300.0
YEARS = [1983, 1988, 1997, 1998, 2005, 2010, 2015]
GAMMAS = [0.9, 1.0, 1.05, 1.1, 1.2]
W = 0.65
THR, BETA = 0.8, 0.25

from worcap.config import EXT_DIR

t0 = time.time()
store = Store()
alvo, _, tgt_month = load_shared()
df = pd.read_parquet(EXT_DIR / "features_shared.parquet")
n34 = df["nino34"].reindex(store.t_idx)
Xraw = production_cols(df).reindex(pd.DatetimeIndex(alvo.time.values)) \
        .fillna(0.0).values.astype(np.float64)
Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
print(f"setup {time.time()-t0:.0f}s", flush=True)

P = {1: {}, 2: {}}
CLIM, PM = {}, {}
for yr in YEARS:
    m = fit_pixel_ridge(store, alvo, Xraw, tgt_month, LOOKUP[yr],
                        LAM1, LAM2, use_nmme=False)
    m["n34"] = n34
    CLIM[yr] = m["clim"].values
    for g in (1, 2):
        P[g][yr] = eval_year_preds(store, alvo, m, yr, g)
    PM[yr] = phase_composites(alvo, LOOKUP[yr], n34, warm=THR, cold=-THR)
    print(f"{yr}: preds ok | {time.time()-t0:.0f}s", flush=True)


def eval_gamma(years, gap, gamma, beta=0.0, phase_sel=None):
    """clip(clim + gamma*(blend - clim) + off) — ordem damp->enso."""
    e = n = 0.0
    for yr in years:
        for d in P[gap][yr]:
            ph = phase_of(d["n34"], THR, -THR)
            if phase_sel is not None and ph != phase_sel:
                continue
            p1 = np.clip(d["base"], 0, None)
            p2 = np.clip(d["base"] + d["corr"], 0, None)
            b = W * p1 + (1 - W) * p2
            off = enso_offset(PM[yr], CLIM[yr], beta,
                              ph)[d["tm"] - 1].ravel()
            yh = np.clip(d["clim"] + gamma * (b - d["clim"]) + off, 0, None)
            y = d["y"]
            e += np.nansum((yh - y) ** 2)
            n += np.isfinite(y).sum()
    return np.sqrt(e / n)


print("\n=== GAMMA por FASE (w=0.65, sem offset) ===")
for g in (1, 2):
    for ph, nm in [(1, "QUENTE"), (0, "NEUTRO"), (-1, "FRIO")]:
        row = [eval_gamma(YEARS, g, ga, phase_sel=ph) for ga in GAMMAS]
        print(f"gap{g} {nm:7s} | " + " ".join(
            f"{ga:.2f}:{v:.4f}" for ga, v in zip(GAMMAS, row)), flush=True)

print("\n=== COMBINADO offset(b=0.25) x gamma — ordem damp->enso ===")
for g in (1, 2):
    for ga in GAMMAS:
        v0 = eval_gamma(YEARS, g, ga, beta=0.0)
        v1 = eval_gamma(YEARS, g, ga, beta=BETA)
        print(f"gap{g} g={ga:.2f} | sem off {v0:.4f} | "
              f"com off {v1:.4f} | d={v1-v0:+.4f}", flush=True)

print("\n=== gamma por ANO (w=0.65, beta=0) ===")
print("ano | " + " ".join(f"g{ga:.2f}" for ga in GAMMAS))
for yr in YEARS:
    for g in (1, 2):
        row = [eval_gamma([yr], g, ga) for ga in GAMMAS]
        print(f"{yr} gap{g} | " + " ".join(f"{v:.4f}" for v in row),
              flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
