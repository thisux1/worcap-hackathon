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
assert hasattr(_enso, "phase_composites"), \
    "dataset worcap-assets2 desatualizado (falta worcap/enso.py)"
from worcap.gbm_data import Store
from worcap.pixel_ridge import load_shared, fit_pixel_ridge, eval_year_preds
from worcap.folds import loyo_folds
from worcap.features import production_cols
from worcap.enso import phase_of, phase_composites, enso_offset

LOOKUP = {f["test_year"]: f["train_years"] for f in loyo_folds()}

LAM1, LAM2 = 1000.0, 300.0
YEARS = [1983, 1988, 1992, 1997, 1998, 2005, 2010, 2015, 2016]
TRANS = [1997, 1998]                    # anos de transicao ENSO forte
NEUT = [y for y in YEARS if y not in TRANS]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
THRS = [0.4, 0.5, 0.8]                  # limiar |nino34| p/ fase warm/cold

from worcap.config import EXT_DIR

t0 = time.time()
store = Store()
alvo, _, tgt_month = load_shared()
df = pd.read_parquet(EXT_DIR / "features_shared.parquet")
# nino34 indexado por mes de ORIGEM (causal: mesmo vintage de Xn[i])
n34 = df["nino34"].reindex(store.t_idx)
Xraw = production_cols(df).reindex(pd.DatetimeIndex(alvo.time.values)) \
        .fillna(0.0).values.astype(np.float64)
Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
print(f"setup {time.time()-t0:.0f}s | X {Xraw.shape} "
      f"| n34 NaN {int(np.isnan(n34.values).sum())}", flush=True)

# --- 1 fit por ano; preds mensais gap1+gap2; compositos por (ano, thr) ---
P = {1: {}, 2: {}}
CLIM, PM = {}, {}
for yr in YEARS:
    m = fit_pixel_ridge(store, alvo, Xraw, tgt_month, LOOKUP[yr],
                        LAM1, LAM2, use_nmme=False)
    m["n34"] = n34
    CLIM[yr] = m["clim"].values
    for g in (1, 2):
        P[g][yr] = eval_year_preds(store, alvo, m, yr, g)
    for thr in THRS:
        PM[(yr, thr)] = phase_composites(alvo, LOOKUP[yr], n34,
                                         warm=thr, cold=-thr)
    nph = {p: int(np.isfinite(PM[(yr, 0.5)][p][0]).sum()) for p in (1, -1)}
    print(f"{yr}: preds ok | pix c/ comp. jan N={nph[1]} S={nph[-1]} "
          f"| {time.time()-t0:.0f}s", flush=True)


def rmse_enso(years, gap, beta, thr, s2=True):
    """RMSE de clip(base + beta*(clim_fase[tm]-clim[tm]) [+ corr]).

    Fase = phase_of(nino34 do mes-origem daquele alvo). Compositos do
    proprio fold (PM[(yr,thr)]) => pos-processamento LOYO-safe.
    """
    e = n = 0.0
    for yr in years:
        pm = PM[(yr, thr)]
        for d in P[gap][yr]:
            ph = phase_of(d["n34"], thr, -thr)
            off = enso_offset(pm, CLIM[yr], beta, ph)[d["tm"] - 1].ravel()
            yh = np.clip(d["base"] + off + (d["corr"] if s2 else 0.0),
                         0, None)
            y = d["y"]
            e += np.nansum((yh - y) ** 2)
            n += np.isfinite(y).sum()
    return np.sqrt(e / n)


for g in (1, 2):
    print(f"\n=== gap{g} | s1 apenas: clip(base+off) | overall {len(YEARS)}y ===")
    print("beta\\thr " + "  ".join(f"{t:.1f}" for t in THRS))
    for b in BETAS:
        print(f"{b:.2f} | " + " ".join(
            f"{rmse_enso(YEARS, g, b, t, s2=False):.4f}" for t in THRS),
            flush=True)
    print(f"=== gap{g} | s1+s2: clip(base+off+corr) | overall ===")
    print("beta\\thr " + "  ".join(f"{t:.1f}" for t in THRS))
    for b in BETAS:
        print(f"{b:.2f} | " + " ".join(
            f"{rmse_enso(YEARS, g, b, t):.4f}" for t in THRS), flush=True)

print("\n=== detalhe s1+s2: por ano | trans(97/98) | neutros ===")
hdr = "gap thr beta | " + " ".join(f"{y}" for y in YEARS) \
      + " | TRANS | NEUT"
print(hdr)
for g in (1, 2):
    for t in THRS:
        for b in BETAS:
            py = [rmse_enso([y], g, b, t) for y in YEARS]
            tr = rmse_enso(TRANS, g, b, t)
            ne = rmse_enso(NEUT, g, b, t)
            print(f"{g} {t:.1f} {b:.2f} | "
                  + " ".join(f"{v:.4f}" for v in py)
                  + f" | {tr:.4f} | {ne:.4f}", flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
