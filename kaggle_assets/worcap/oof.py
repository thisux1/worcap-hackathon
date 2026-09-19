"""OOF prediction store: parquet (time, lat, lon, pred, fold, via)."""
import numpy as np
import pandas as pd
import xarray as xr

from .config import OOF_DIR


def save_oof(via: str, fold: int, pred: xr.DataArray):
    OOF_DIR.mkdir(exist_ok=True)
    df = (
        pred.to_dataframe(name="pred").reset_index()
        .assign(fold=fold, via=via)
    )
    df[["lat", "lon"]] = df[["lat", "lon"]].astype(np.float32)
    df["pred"] = df["pred"].astype(np.float32)
    out = OOF_DIR / f"{via}__fold{fold}.parquet"
    df.to_parquet(out)
    return out


def load_oof(via: str = None) -> pd.DataFrame:
    files = sorted(OOF_DIR.glob(f"{via or '*'}__fold*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
