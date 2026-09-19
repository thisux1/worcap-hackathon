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
assert hasattr(_pr, "eval_year_rec"), "dataset worcap-assets2 desatualizado"
from worcap.gbm_data import Store
from worcap.pixel_ridge import (load_shared, fit_pixel_ridge, eval_year,
                                eval_year_preds)
from worcap.folds import loyo_folds
from worcap.features import production_cols

LOOKUP = {f["test_year"]: f["train_years"] for f in loyo_folds()}

LAM1, LAM2 = 1000.0, 300.0
YEARS = [1983, 1988, 1997, 1998, 2005, 2010, 2015]
GS = [0.9, 1.0, 1.05, 1.1, 1.2]          # inclui INFLACAO (g > 1)
WS = [0.0, 0.65]                          # v2 puro | blend 65/35 s1 (prod)
VARIANTS = [                             # flags novas (default OFF = base)
    ("base",        {}),
    ("wins4",       {"winsorize_sigma": 4.0}),
    ("x2std",       {"x2_std": True}),
    ("wins4+x2std", {"winsorize_sigma": 4.0, "x2_std": True}),
]

from worcap.config import EXT_DIR

t0 = time.time()
store = Store()
alvo, _, tgt_month = load_shared()
df = production_cols(pd.read_parquet(EXT_DIR / "features_shared.parquet"))
Xraw = df.reindex(pd.DatetimeIndex(alvo.time.values)).fillna(0.0) \
        .values.astype(np.float64)
Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
print(f"setup {time.time()-t0:.0f}s | X {Xraw.shape}", flush=True)

res = {}                                    # (variant, yr) -> (r1, r2) x gap
gsse = {v: {g: np.zeros((len(WS), len(GS))) for g in (1, 2)}
        for v, _ in VARIANTS}
gn = {v: {1: 0.0, 2: 0.0} for v, _ in VARIANTS}

for vname, kw in VARIANTS:
    for yr in YEARS:
        m = fit_pixel_ridge(store, alvo, Xraw, tgt_month, LOOKUP[yr],
                            LAM1, LAM2, use_nmme=False, **kw)
        res[(vname, yr)] = (eval_year(store, alvo, m, yr, 1),
                            eval_year(store, alvo, m, yr, 2))
        r = res[(vname, yr)]
        print(f"{vname:12s} {yr} | gap1 s1 {r[0][0]:.4f} s12 {r[0][1]:.4f}"
              f" | gap2 s1 {r[1][0]:.4f} s12 {r[1][1]:.4f}"
              f" | {time.time()-t0:.0f}s", flush=True)
        for g in (1, 2):
            for mo in eval_year_preds(store, alvo, m, yr, g):
                p1 = np.clip(mo["base"], 0, None)
                p2 = np.clip(mo["base"] + mo["corr"], 0, None)
                y, clim = mo["y"], mo["clim"]
                gn[vname][g] += np.isfinite(y).sum()
                for wi, w in enumerate(WS):
                    b = w * p1 + (1 - w) * p2
                    for gi, gam in enumerate(GS):
                        d = np.clip(clim + gam * (b - clim), 0, None)
                        gsse[vname][g][wi, gi] += np.nansum((d - y) ** 2)

print("\n=== MEDIA RMSE por variante (s1 / s1+2) ===")
best, best_score = None, np.inf
for vname, _ in VARIANTS:
    m1 = np.mean([res[(vname, y)][0] for y in YEARS], axis=0)
    m2 = np.mean([res[(vname, y)][1] for y in YEARS], axis=0)
    score = (m1[1] + m2[1]) / 2
    if score < best_score:
        best, best_score = vname, score
    print(f"{vname:12s} | gap1 {m1[0]:.4f}/{m1[1]:.4f}"
          f" | gap2 {m2[0]:.4f}/{m2[1]:.4f} | s12 media {score:.4f}")

print(f"\n=== GAMMA GRID — melhor variante: {best} ===")
for g in (1, 2):
    n = gn[best][g]
    print(f"-- gap{g} (n={int(n)}) --")
    print("w\\gamma  " + "   ".join(f"{gam:.2f}" for gam in GS))
    for wi, w in enumerate(WS):
        row = [np.sqrt(gsse[best][g][wi, gi] / n) for gi in range(len(GS))]
        print(f"{w:.2f}   | " + " ".join(f"{v:.4f}" for v in row))

print("\n=== best gamma por variante (w=0.65, gap1/gap2) ===")
wi = WS.index(0.65)
for vname, _ in VARIANTS:
    line = []
    for g in (1, 2):
        n = gn[vname][g]
        gi = int(np.argmin(gsse[vname][g][wi]))
        line.append(f"g{GS[gi]:.2f}:{np.sqrt(gsse[vname][g][wi, gi]/n):.4f}")
    print(f"{vname:12s} | gap1 {line[0]} | gap2 {line[1]}")
print(f"done {time.time()-t0:.0f}s", flush=True)
