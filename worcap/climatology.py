"""Climatologia por (mes-calendario, pixel): media + tendencia linear opcional.

V0 e componente central: com RMSE absoluto, clim e sigma entram sem atenuacao
em y_hat = clim + sigma * z_hat. Sempre fitar apenas com train_years do fold.
"""
import numpy as np
import xarray as xr


def fit_climatology(tp: xr.DataArray, years, trend: bool = False):
    """Retorna (clim, sigma) com dims (month, lat, lon).

    clim[m] = a + b*(year - y0) se trend=True, senao media simples dos anos dados.
    sigma[m] = desvio-padrao dos residuos (mesmo conjunto).
    """
    t = tp.sel(time=tp.time.dt.year.isin(list(years)))
    month = t.time.dt.month
    clim = t.groupby(month).mean("time")
    resid = t.groupby(month) - clim
    sigma = resid.groupby(month).std("time")

    if trend:
        # refit com tendencia linear por (mes, pixel): y = a + b*(ano - y0)
        out_c = np.zeros((12, tp.sizes["lat"], tp.sizes["lon"]), np.float32)
        for m in range(1, 13):
            tm = t.sel(time=t.time.dt.month == m)
            yy = tm.time.dt.year.values.astype(np.float64)
            y0 = yy.mean()
            X = np.stack([np.ones_like(yy), yy - y0], 1)
            Y = tm.values  # (n, lat, lon)
            beta = np.linalg.lstsq(X, Y.reshape(len(yy), -1), rcond=None)[0]
            out_c[m - 1] = beta[0].reshape(tp.sizes["lat"], tp.sizes["lon"])
        clim = xr.DataArray(
            out_c,
            dims=("month", "lat", "lon"),
            coords={"month": np.arange(1, 13), "lat": tp.lat, "lon": tp.lon},
        )
    return clim, sigma


def climatology_predict(clim: xr.DataArray, times) -> xr.DataArray:
    """Expande clim (month,lat,lon) para os meses-alvo dados (datetime64)."""
    months = xr.DataArray(times).dt.month
    return clim.sel(month=months.values).assign_coords(time=("month", times)).rename(
        {"month": "time"}).transpose("time", "lat", "lon")


def zscore(tp: xr.DataArray, clim: xr.DataArray, sigma: xr.DataArray) -> xr.DataArray:
    """Anomalia padronizada z = (y - clim_mes) / sigma_mes."""
    m = tp.time.dt.month
    c = clim.sel(month=m.values).rename({"month": "time"}).assign_coords(time=tp.time)
    s = sigma.sel(month=m.values).rename({"month": "time"}).assign_coords(time=tp.time)
    return (tp - c) / s
