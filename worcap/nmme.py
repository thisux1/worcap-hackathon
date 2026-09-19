"""NMME/C3S: previsoes dinamicas debiased como features (V3-lite).

Fonte: C3S seasonal-monthly-single-levels, centro ncep (CFSv2, system 2),
leadtime_month=1 -> init no mes S preve o mes S+1 (mesma indexacao por mes
de ORIGEM da tabela compartilhada). Causal: init emitido ~dia 1-8 de S.

Cobertura real: hindcast 1993-2016 (28 membros) + realtime 2019-10..2024-12
(124 membros). Gap 2017..2019-09 e pre-1993 -> anomalia 0 (neutro).

Debias: anomalia = fcst - clim_modelo(mes-init, excluindo anos do fold).
Fluxo: build_raw_grid() regrida ensemble-mean p/ grade ERA5 uma vez;
as anomalias se calculam por fold a partir do arquivo bruto (sem leakage).
"""
import numpy as np
import pandas as pd
import xarray as xr

from .config import DATA, TP_FILE, EXT_DIR
from .features import REGIONS, region_mean

NMME_DIR = EXT_DIR / "nmme"
RAW_GRID = NMME_DIR / "nmme_raw_grid.nc"
RAW_GRID_FLAT = EXT_DIR / "nmme_raw_grid.nc"   # fallback p/ mount flat
HIND_MAX_YEAR = 2016   # inits <=2016 entram na clim do modelo (hindcast)


def _files():
    return sorted(NMME_DIR.glob("*_lead1.nc"))


def _model_of(path):
    return path.name.split("_")[0]          # ncep / ecmwf / ...


def _norm(da: xr.DataArray) -> xr.DataArray:
    """Uniformiza dims: (init, lat, lon), mm/dia."""
    if "forecastMonth" in da.dims:
        da = da.squeeze("forecastMonth")
    tdim = [d for d in da.dims if "time" in d][0]
    return (da.rename({tdim: "init"}) * 86400.0 * 1000.0)  # m/s -> mm/dia


def build_raw_grid(chunk=48):
    """Regrida ensemble-mean de CADA modelo p/ grade ERA5 -> raw_grid.nc.

    Dataset com uma var por centro (ncep, ecmwf, ...), dims (init,lat,lon).
    """
    ref = xr.open_dataset(DATA / TP_FILE)["tp"]
    out = {}
    for f in _files():
        model = _model_of(f)
        ds = xr.open_dataset(f)
        da = _norm(ds["tprate"].mean("number"))
        n = da.sizes["init"]
        blks = []
        for i0 in range(0, n, chunk):
            blk = da.isel(init=slice(i0, i0 + chunk)).load()
            blks.append(blk.interp(latitude=ref.lat,
                                   longitude=ref.lon).astype(np.float32))
        ds.close()
        part = xr.concat(blks, dim="init")
        out[model] = part if model not in out else xr.concat(
            [out[model], part], dim="init")
    dsr = xr.Dataset({m: v.sortby("init") for m, v in out.items()})
    dsr.to_netcdf(RAW_GRID)
    return dsr


def load_raw(model=None):
    """Raw (init,lat,lon) de um modelo (ou media dos modelos, model=None)."""
    path = RAW_GRID if RAW_GRID.exists() else RAW_GRID_FLAT
    if not path.exists():
        build_raw_grid()
        path = RAW_GRID
    ds = xr.open_dataset(path).load()
    if model is not None:
        return ds[model]
    return ds.to_array("model").mean("model")


def anom_on_grid(raw: xr.DataArray, exclude_years=()) -> xr.DataArray:
    """Anomalia debiased por mes-init, excluindo anos do fold (LOYO-safe)."""
    t = pd.DatetimeIndex(raw.init.values)
    keep = np.array([(y <= HIND_MAX_YEAR and y not in exclude_years)
                     for y in t.year])
    sub = raw.isel(init=keep)
    mc = sub.groupby(sub.init.dt.month).mean("init")
    return raw.groupby(raw.init.dt.month) - mc


def region_table(exclude_years=(), model=None) -> pd.DataFrame:
    """init-mes x {nmme_<reg>_an} (anomalia debiased regional)."""
    an = anom_on_grid(load_raw(model), exclude_years)
    rows = {}
    for rname, box in REGIONS.items():
        rows[f"nmme_{rname}_an"] = np.nan_to_num(
            region_mean(an, box).values)
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(an.init.values))
    df.index.name = "time"
    return df


def anom_field(raw: xr.DataArray, init_month, exclude_years=()):
    """Anomalia (lat, lon) de um init; None se o init nao existir."""
    t = pd.DatetimeIndex(raw.init.values)
    hits = np.where(t == pd.Timestamp(init_month))[0]
    if len(hits) == 0:
        return None
    return np.nan_to_num(
        anom_on_grid(raw, exclude_years).isel(init=hits[0]).values)
