"""Inflacao gamma SOMENTE em meses-alvo de fase ENSO quente (nino34 origem >= thr).

OOF (kernel gphase): g>1 melhora monotonico em meses quentes (-0.009 em g=1.2),
e' ~neutro/piora em neutros e frios. Uso: private-tilt 2024 (decay El Nino),
onde os meses quentes jan-mai sao analogos a 1983/1998.

Uso: python scripts/gamma_warm.py in.csv out.csv [gamma] [thr]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from worcap.submission import validate_submission
from worcap.config import DATA, EXT_DIR, TP_FILE


def main():
    inp, out = sys.argv[1], sys.argv[2]
    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 1.10
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8

    tp = xr.open_dataset(DATA / TP_FILE)["tp"]
    clim = tp.groupby("time.month").mean("time")
    n34 = pd.read_parquet(EXT_DIR / "features_shared.parquet")["nino34"]

    df = pd.read_csv(inp)
    ids = df["id"].str.split("_", expand=True)
    year = ids[0].astype(int).values
    month = ids[1].astype(int).values
    lat_i = ((ids[2].astype(float) + 60.0) / 0.25).round().astype(int).values
    lon_i = ((ids[3].astype(float) + 90.0) / 0.25).round().astype(int).values

    # mes de origem = alvo - 1 mes
    orig = pd.to_datetime(dict(year=year, month=month, day=1)) \
        - pd.offsets.MonthBegin(1)
    n34v = n34.reindex(pd.DatetimeIndex(orig)).fillna(0.0).values
    warm = n34v >= thr

    c = clim.values[month - 1, lat_i, lon_i]
    p = df["tp_mm_day"].values
    p_w = np.clip(c + gamma * (p - c), 0, None)
    df["tp_mm_day"] = np.where(warm, p_w, p)
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"{out} | linhas quentes infladas: {int(warm.sum())}")
    print(validate_submission(out))


if __name__ == "__main__":
    main()
