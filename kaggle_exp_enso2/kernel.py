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
assert hasattr(_pr, "eval_year_rec"), "dataset worcap-assets2 desatualizado"
assert hasattr(_enso, "phase_composites_2way"), \
    "dataset worcap-assets2 desatualizado (falta enso 2-way)"
from worcap.gbm_data import Store
from worcap.pixel_ridge import load_shared, fit_pixel_ridge, eval_year_preds
from worcap.folds import loyo_folds
from worcap.features import production_cols
from worcap.enso import (phase_of, phase_composites, enso_offset,
                         phase_composites_2way, enso_offset_2way,
                         assign_bins)

LOOKUP = {f["test_year"]: f["train_years"] for f in loyo_folds()}

LAM1, LAM2 = 1000.0, 300.0
YEARS = [1983, 1988, 1992, 1997, 1998, 2005, 2010, 2015, 2016]
TRANS = [1997, 1998]                    # anos de transicao ENSO forte
NEUT = [y for y in YEARS if y not in TRANS]
BETAS = [0.0, 0.15, 0.25, 0.35]
THR1 = 0.8                              # limiar |nino34| p/ fase1
K2W = 5.0                               # shrinkage da celula p/ fase1
IDX2 = ["tna", "atl3", "nino12", "ep_cp"]
BMODES = ["sign", "tercile"]            # bins do 2o indice
LABELS = ["1way"] + [f"{n}:{b}" for n in IDX2 for b in BMODES]

from worcap.config import EXT_DIR

t0 = time.time()
store = Store()
alvo, _, tgt_month = load_shared()
df = pd.read_parquet(EXT_DIR / "features_shared.parquet")
# indices por mes de ORIGEM (causal: mesmo vintage de Xn[i])
n34 = df["nino34"].reindex(store.t_idx)
idx2 = {n: df[n].reindex(store.t_idx) for n in IDX2}
Xraw = production_cols(df).reindex(pd.DatetimeIndex(alvo.time.values)) \
        .fillna(0.0).values.astype(np.float64)
Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
print(f"setup {time.time()-t0:.0f}s | X {Xraw.shape} "
      f"| n34 NaN {int(np.isnan(n34.values).sum())}", flush=True)

# --- 1 fit por ano; preds mensais gap1+gap2; compositos 1-way por ano ---
P = {1: {}, 2: {}}
CLIM, PM1 = {}, {}
for yr in YEARS:
    m = fit_pixel_ridge(store, alvo, Xraw, tgt_month, LOOKUP[yr],
                        LAM1, LAM2, use_nmme=False)
    m["n34"] = n34
    CLIM[yr] = m["clim"].values
    for g in (1, 2):
        P[g][yr] = eval_year_preds(store, alvo, m, yr, g)
    PM1[yr] = phase_composites(alvo, LOOKUP[yr], n34,
                               warm=THR1, cold=-THR1)
    print(f"{yr}: preds ok | {time.time()-t0:.0f}s", flush=True)

# --- acumula erro quadratico por (variante, gap, ano, beta) -------------
# offset e' linear em beta: calcula off unitario (beta=1) e escala.
E, Nc = {}, {}


def accum(label, gap, yr, d, off_u):
    y = d["y"]
    nn = np.isfinite(y)
    for b in BETAS:
        yh = np.clip(d["base"] + b * off_u + d["corr"], 0, None)
        k = (label, gap, yr, b)
        E[k] = E.get(k, 0.0) + float(np.nansum((yh - y) ** 2))
        Nc[k] = Nc.get(k, 0) + int(nn.sum())


# baseline 1-way (mesmo sweep de beta; beta=0 = baseline puro)
for yr in YEARS:
    for g in (1, 2):
        for d in P[g][yr]:
            ph = phase_of(d["n34"], THR1, -THR1)
            off = enso_offset(PM1[yr], CLIM[yr], 1.0, ph)
            accum("1way", g, yr, d, off[d["tm"] - 1].ravel())
print(f"1way ok | {time.time()-t0:.0f}s", flush=True)

# variantes 2-way: compositos refitados por (variante, ano) -> memoria ~1 mapa
for name in IDX2:
    x2s = idx2[name]
    x2v = x2s.values
    for bm in BMODES:
        lab = f"{name}:{bm}"
        for yr in YEARS:
            pm2 = phase_composites_2way(alvo, LOOKUP[yr], n34, x2s,
                                        thr1=THR1, bins2=bm, k=K2W)
            nb = {kb: int(pm2["n"][kb].sum()) for kb in pm2["n"]
                  if kb[0] != 0}
            for g in (1, 2):
                for d in P[g][yr]:
                    ph = phase_of(d["n34"], THR1, -THR1)
                    bn = int(assign_bins([x2v[d["i"]]],
                                         pm2["edges"])[0])
                    off = enso_offset_2way(pm2, CLIM[yr], 1.0, ph,
                                           None if bn < 0 else bn)
                    accum(lab, g, yr, d, off[d["tm"] - 1].ravel())
            print(f"{lab} {yr}: edges={np.round(pm2['edges'], 3)} "
                  f"n={nb} | {time.time()-t0:.0f}s", flush=True)


def rmse(label, gap, years, beta):
    e = n = 0.0
    for yr in years:
        k = (label, gap, yr, beta)
        e += E.get(k, 0.0)
        n += Nc.get(k, 0)
    return np.sqrt(e / n)


for g in (1, 2):
    print(f"\n=== gap{g} | s1+s2: clip(base + beta*off + corr) "
          f"| thr1={THR1} k={K2W} ===")
    print("variant      beta | " + " ".join(f"{y}" for y in YEARS)
          + " |   ALL  | TRANS  |  NEUT")
    for lab in LABELS:
        for b in BETAS:
            py = [rmse(lab, g, [y], b) for y in YEARS]
            print(f"{lab:12s} {b:.2f} | "
                  + " ".join(f"{v:.4f}" for v in py)
                  + f" | {rmse(lab, g, YEARS, b):.4f}"
                  + f" | {rmse(lab, g, TRANS, b):.4f}"
                  + f" | {rmse(lab, g, NEUT, b):.4f}", flush=True)
    print(f"--- gap{g} dRMSE vs beta=0 (ALL e 2016) ---")
    for lab in LABELS:
        row = " ".join(
            f"{b:.2f}:{rmse(lab, g, YEARS, b) - rmse(lab, g, YEARS, 0.0):+.4f}"
            for b in BETAS[1:])
        d16 = {b: rmse(lab, g, [2016], b) - rmse(lab, g, [2016], 0.0)
               for b in BETAS[1:]}
        print(f"{lab:12s} ALL {row} | 2016 "
              + " ".join(f"{b:.2f}:{v:+.4f}" for b, v in d16.items()),
              flush=True)

print(f"done {time.time()-t0:.0f}s", flush=True)
