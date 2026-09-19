"""Damping: encolhe a componente anomala da previsao em direcao a climatologia.

pred_damped = clip(clim_mes + gamma * (pred - clim_mes), 0)
gamma=0.80 selecionado no OOF store (v05_ridge): RMSE 1.7406 vs 1.7442 (g=1).

Uso: python scripts/damp.py in.csv out.csv [gamma]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from worcap.submission import validate_submission
from worcap.config import DATA, TP_FILE


def main():
    inp, out = sys.argv[1], sys.argv[2]
    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.80
    tp = xr.open_dataset(DATA / TP_FILE)["tp"]
    clim = tp.groupby("time.month").mean("time")   # (12, 301, 261)

    df = pd.read_csv(inp)
    ids = df["id"].str.split("_", expand=True)
    month = ids[1].astype(int).values
    lat_i = ((ids[2].astype(float) + 60.0) / 0.25).round().astype(int).values
    lon_i = ((ids[3].astype(float) + 90.0) / 0.25).round().astype(int).values
    assert lat_i.min() >= 0 and lat_i.max() <= 300
    assert lon_i.min() >= 0 and lon_i.max() <= 260

    c = clim.values[month - 1, lat_i, lon_i]
    p = df["tp_mm_day"].values
    df["tp_mm_day"] = np.clip(c + gamma * (p - c), 0, None)
    df.to_csv(out, index=False, float_format="%.6f")
    print(out, validate_submission(out))


if __name__ == "__main__":
    main()
