"""Indices climaticos computados do ERSSTv5 (data_ext/ersst_v5_sst_mnmean.nc).

Anomalias contra a climatologia mensal de 1940-2022 (mesmo periodo do treino).
Saida: data_ext/indices.parquet — colunas por indice, index = mes (datetime).
"""
import numpy as np
import pandas as pd
import xarray as xr

from .config import EXT_DIR

ERSST = EXT_DIR / "ersst_v5_sst_mnmean.nc"
MSLP_SO = EXT_DIR / "era5_mslp_southocean.nc"
CLIM_YEARS = (1940, 2022)

# caixas (lon em graus E 0-360, lat): nome -> (lonW, lonE, latS, latN)
BOXES = {
    "nino12": (270, 280, -10, 0),
    "nino3":  (210, 270, -5, 5),
    "nino34": (190, 240, -5, 5),
    "nino4":  (160, 210, -5, 5),
    "tna":    (305, 345, 5.5, 23.5),      # Atlantico N tropical
    "tsa":    (330, 370, -20, 0),         # Atlantico S tropical (30W-10E)
    "atl3":   (340, 360, -3, 3),          # Nino Atlantico
    "tio":    (40, 110, -20, 20),         # Indico tropical de bacia
    "saod_n": (330, 370, -25, -10),       # polo norte do dipolo do Atl. Sul (costa africana)
    "saod_s": (300, 330, -45, -35),       # polo sul (Atl. SW)
    "pmm_n":  (180, 265, 10, 30),         # Pacifico subtropical N (proxy PMM)
    "npac":   (150, 210, 20, 60),         # Pacifico Norte (proxy PDO)
}


def box_mean(sst: xr.DataArray, lonW, lonE, latS, latN) -> xr.DataArray:
    """Media ponderada por cos(lat) na caixa; suporta lonE>360 (cruza 0)."""
    lat_slice = slice(latN, latS) if sst.lat.values[0] > sst.lat.values[-1] else slice(latS, latN)
    sel = sst.sel(lat=lat_slice)
    if lonE > 360:
        a = sel.sel(lon=slice(lonW, 360))
        b = sel.sel(lon=slice(0, lonE - 360))
        sel = xr.concat([a, b], dim="lon")
    else:
        sel = sel.sel(lon=slice(lonW, lonE))
    w = np.cos(np.deg2rad(sel.lat))
    return sel.weighted(w).mean(("lat", "lon"), skipna=True)


def compute_sam() -> pd.Series:
    """SAM-proxy (Gong & Wang): zonal-mean MSLP*40S - MSLP*65S, ambos z-scored.

    Fonte: ERA5 monthly MSLP banda -35..-70 (data_ext/era5_mslp_southocean.nc).
    O dominio oficial so vai ate -60 -> SAM canonico exige download externo.
    """
    ds = xr.open_dataset(MSLP_SO)
    v = "msl" if "msl" in ds else list(ds.data_vars)[0]
    p = ds[v].load()
    z = p.mean("longitude") if "longitude" in p.dims else p.mean("lon")
    latc = "latitude" if "latitude" in z.coords else "lat"
    tc = "valid_time" if "valid_time" in z.dims else "time"
    z = z.rename({tc: "time"})
    t = pd.DatetimeIndex(z.time.values)
    def band(lat0):
        sel = z.sel({latc: slice(lat0 - 2.5, lat0 + 2.5)} if
                    z[latc].values[0] < z[latc].values[-1]
                    else {latc: slice(lat0 + 2.5, lat0 - 2.5)})
        return sel.mean(latc)
    def zscore(s):
        clim = s.sel(time=slice("1940", "2022")).groupby("time.month").mean("time")
        an = s.groupby("time.month") - clim
        sd = an.sel(time=slice("1940", "2022")).std("time")
        return an / sd
    sam = zscore(band(-40)) - zscore(band(-65))
    return pd.Series(sam.values, index=t, name="sam")


def compute_pdo(sst: xr.DataArray) -> pd.Series:
    """PDO classica: PC1 da anomalia de SST no Pacifico Norte (20-70N, 110E-100W),
    apos remover a media global por timestep (convencao Mantua/Deser).
    Orientacao: corr positiva com nino34 (PDO+ ~ fase quente tipo El Nino)."""
    lon = sst.lon
    lonE = lon.where(lon <= 180, lon - 360)
    sst2 = sst.assign_coords(lon=lonE)
    lat_n = sst2.lat.where(sst2.lat >= 20, drop=True)
    pac = sst2.sel(lat=lat_n).where(
        (sst2.lon >= 110) | (sst2.lon <= -100), drop=True)
    pac = pac.sel(lat=slice(70, 20)) if pac.lat.values[0] > pac.lat.values[-1] \
        else pac.sel(lat=slice(20, 70))
    base = sst2.sel(time=slice("1940", "2022"))
    clim = base.groupby("time.month").mean("time")
    an = sst2.groupby("time.month") - clim          # anomalia global
    gmean = an.weighted(np.cos(np.deg2rad(an.lat))).mean(("lat", "lon"))
    pac_an = an.sel(lat=pac.lat).sel(lon=pac.lon) - gmean
    w = np.sqrt(np.cos(np.deg2rad(pac.lat)))
    X = np.nan_to_num((pac_an * w).values.reshape(len(pac.time), -1))
    t = pd.DatetimeIndex(pac.time.values)
    fit = t <= "2022-12-31"
    # SVD/statistics fitados so em <=2022: projeta 2023-24 sobre loadings congelados
    U, S, Vt = np.linalg.svd(X[fit], full_matrices=False)
    pc = np.empty(len(t))
    pc[fit] = U[:, 0] * S[0]
    pc[~fit] = X[~fit] @ Vt[0]
    n34 = box_mean(sst, *BOXES["nino34"])
    n34_an = (n34.groupby("time.month")
              - n34.sel(time=slice("1940", "2022"))
                  .groupby("time.month").mean("time")).reindex(time=t)
    if np.corrcoef(pc[fit], n34_an.values[fit])[0, 1] < 0:
        pc = -pc
    pc = (pc - pc[fit].mean()) / pc[fit].std()
    return pd.Series(pc, index=t, name="pdo")


def compute_indices() -> pd.DataFrame:
    ds = xr.open_dataset(ERSST).sst.load()          # (time, lat, lon) 1854+
    ds = ds.where(ds > -50)                          # -1.8 fill de gelo/terra -> NaN
    time = pd.DatetimeIndex(ds.time.values)
    base = ds.sel(time=slice(f"{CLIM_YEARS[0]}", f"{CLIM_YEARS[1]}-12"))

    out = {}
    for name, box in BOXES.items():
        ts = box_mean(ds, *box)
        clim_m = box_mean(base, *box).groupby("time.month").mean("time")
        anom = ts.groupby("time.month") - clim_m
        out[name] = anom.values

    df = pd.DataFrame(out, index=time)
    # derivados
    df["amm"] = df["tna"] - df["tsa"]                # gradiente meridional
    df["saod"] = df["saod_n"] - df["saod_s"]
    df["ep_cp"] = df["nino3"] - df["nino4"]          # diversidade EP/CP
    df["pdo_x_nino34"] = df["npac"] * df["nino34"]   # proxy PDO x ENSO
    return df.drop(columns=["saod_n", "saod_s", "pmm_n"]) if False else df


def main():
    df = compute_indices()
    ds = xr.open_dataset(ERSST).sst
    df["pdo"] = compute_pdo(ds).reindex(df.index)
    df["pdo_x_nino34"] = df["pdo"] * df["nino34"]   # PDO real x ENSO
    if MSLP_SO.exists():
        df["sam"] = compute_sam().reindex(df.index)
        df["sam_x_nino34"] = df["sam"] * df["nino34"]
    df = df.loc["1940":].astype(np.float32)
    df.to_parquet(EXT_DIR / "indices_ersst.parquet")
    print(df.tail(14).round(2).to_string())
    print("salvo:", EXT_DIR / "indices_ersst.parquet", df.shape)


if __name__ == "__main__":
    main()
