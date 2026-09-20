"""Aplica offset ENSO-condicional a uma submissao existente (pos-hoc, aditivo).

offset[mes_alvo] = beta * (clim_fase[mes_alvo] - clim_full[mes_alvo]),
com fase decidida pelo nino34 do mes de ORIGEM (causal: disponivel em T-1).

Por ser aditivo e compartilhado entre componentes, vale para qualquer blend
de modelos que usem a mesma clim_full (fit sobre os mesmos anos).
Uso: python scripts/enso_offset.py in.csv out.csv [beta] [thr]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from worcap.climatology import fit_climatology
from worcap.config import DATA, EXT_DIR, TP_FILE, TARGET_FILE, TEST_FILE
from worcap.enso import enso_offset, phase_composites, phase_of
from worcap.submission import load_sample_index, validate_submission

YEARS = list(range(1940, 2023))  # all_years=True do submit()


def main(src, out, beta=0.25, thr=0.8):
    tp = xr.open_dataset(DATA / TP_FILE)["tp"]
    alvo = xr.open_dataset(DATA / TARGET_FILE)["tp_alvo"]
    clim_full = fit_climatology(tp, YEARS)[0].values          # (12,lat,lon)
    n34 = pd.read_parquet(EXT_DIR / "features_shared.parquet")["nino34"]
    pm = phase_composites(alvo, YEARS, n34, thr, -thr)

    te = xr.open_dataset(DATA / TEST_FILE)
    T = pd.DatetimeIndex(te.time.values)
    orig = pd.DatetimeIndex(te.time_origem.values)

    sub = pd.read_csv(src)
    ids = load_sample_index()["id"]
    assert sub["id"].equals(ids), "ids divergem do sample"
    v = sub["tp_mm_day"].values.reshape(24, 301, 261).copy()
    for j, t in enumerate(T):
        ph = phase_of(n34.get(pd.Timestamp(orig[j]), np.nan), thr, -thr)
        v[j] += enso_offset(pm, clim_full, beta, ph)[t.month - 1]
    pd.DataFrame({"id": ids,
                  "tp_mm_day": np.clip(v.reshape(-1), 0, None)}
                 ).to_csv(out, index=False, float_format="%.6f")
    print(out, validate_submission(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.25,
         float(sys.argv[4]) if len(sys.argv) > 4 else 0.8)
