"""Writer/validator de submissao: ordem vem do sample_submission, nunca reconstruir ids."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from .config import DATA, SAMPLE_FILE, N_TEST_ROWS


def load_sample_index() -> pd.DataFrame:
    """Le os ids do sample (48 MB) uma vez; cache em parquet (dir gravavel)."""
    cache = Path(os.environ.get("WORCAP_CACHE", "/tmp")) / ".sample_index.parquet"
    if not cache.exists():
        alt = Path("/home/thiago/worcap-hackathon/.sample_index.parquet")
        if alt.exists():
            cache = alt
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_csv(DATA / SAMPLE_FILE, usecols=["id"])
    try:
        df.to_parquet(cache)
    except OSError:
        pass
    return df


def write_submission(pred: xr.DataArray, path, float_fmt="%.6f"):
    """pred: DataArray (time=24, lat=301, lon=261) alinhado a teste_features.time.

    Achatamos em C-order (time, lat, lon) — confirmado contra o sample:
    id i = ano_mes_lat_lon com lat externo, lon interno.
    """
    assert pred.sizes["time"] == 24 and pred.sizes["lat"] == 301 and pred.sizes["lon"] == 261
    vals = np.clip(np.asarray(pred.values, dtype=np.float64), 0.0, None)
    ids = load_sample_index()["id"]
    df = pd.DataFrame({"id": ids, "tp_mm_day": vals.reshape(-1)})
    df.to_csv(path, index=False, float_format=float_fmt)
    return path


def validate_submission(path) -> dict:
    """Checagens: header, n de linhas, ids identicos ao sample, sem NaN, >= 0."""
    df = pd.read_csv(path)
    ids = load_sample_index()["id"]
    ok = {
        "columns_ok": list(df.columns) == ["id", "tp_mm_day"],
        "n_rows": len(df) == N_TEST_ROWS,
        "ids_match": df["id"].equals(ids),
        "no_nan": df["tp_mm_day"].notna().all(),
        "non_neg": (df["tp_mm_day"] >= 0).all(),
    }
    ok["valid"] = all(ok.values())
    return ok
