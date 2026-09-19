"""Features: anomalias LOO, medias por macro-regiao, tabela compartilhada mensal.

Decisoes-chave (ver DECISIONS.md D-007/D-008):
- Anomalias usam climatologia leave-one-out por mes-calendario (sem leakage).
- Features externas restritas a <= fim do mes M-1.
- No teste, precipitacao so existe como tp_ultima_obs (dez/2022) + lag_meses.
"""
import numpy as np
import pandas as pd
import xarray as xr

from .config import DATA, EXT_DIR, FEATURE_FILES, FEATURE_VARS, TEST_FILE, TP_FILE

# macro-regioes sobre a grade oficial (lat -60..15 asc, lon -90..-25 asc)
REGIONS = {
    "amazonia": (-15, 5, -75, -50),
    "neb":      (-15, -2, -45, -34),
    "sudeste":  (-25, -15, -50, -38),
    "sul_sesa": (-35, -22, -60, -48),
    "atl_trop": (-5, 10, -40, -20),       # oceano - ZCIT/ATL3 lado oeste
}

# janelas sazonais p/ interacoes (mes do ALVO -> chave da janela)
SEASON_OF_MONTH = {  # alvo M+1
    1: "djf", 2: "fma", 3: "fma", 4: "mam", 5: "mam", 6: "mjj",
    7: "jja", 8: "jja", 9: "son", 10: "son", 11: "son", 12: "djf",
}
INTERACTIONS = [
    ("nino34", ("son", "djf")), ("amm", ("fma", "mam")),
    ("atl3", ("mjj", "jja")), ("saod", ("djf",)), ("ep_cp", ("son", "djf")),
]


def load_var(name: str) -> xr.DataArray:
    return xr.open_dataset(DATA / FEATURE_FILES[name])[name]


def monthly_clim(da: xr.DataArray) -> np.ndarray:
    """(12, lat, lon) climatologia mensal."""
    return da.groupby("time.month").mean("time").values.astype(np.float32)


def loo_anom(da: xr.DataArray, clim: np.ndarray = None) -> np.ndarray:
    """Anomalia vs climatologia leave-one-out do proprio ano (sem leakage)."""
    if clim is None:
        clim = monthly_clim(da)
    months = da.time.dt.month.values
    data = da.values
    out = np.empty_like(data)
    for m in range(1, 13):
        sel = months == m
        blk = data[sel]
        loo_mean = (blk.sum(0) - blk) / (len(blk) - 1)
        out[sel] = blk - loo_mean
    return out.astype(np.float32)


def region_mean(da: xr.DataArray, box) -> xr.DataArray:
    latS, latN, lonW, lonE = box
    sel = da.sel(lat=slice(latS, latN), lon=slice(lonW, lonE))
    w = np.cos(np.deg2rad(sel.lat))
    return sel.weighted(w).mean(("lat", "lon"))


def build_shared_table() -> pd.DataFrame:
    """Features compartilhadas por mes: indices ERSST + medias regionais de
    anomalia das 9 vars + calendario + interacoes indice x janela.

    Linha do mes m alimenta a previsao do alvo m+1.
    Retorna DataFrame indexado por mes (datetime64), 1940-01..2024-12.
    """
    idx = pd.read_parquet(EXT_DIR / "indices_ersst.parquet")

    # medias regionais de anomalia das 9 vars (LOO), treino + teste
    # atencao: no teste a var do alvo T e o estado de M-1 -> indexar por time_origem
    rows = {}
    test = xr.open_dataset(DATA / TEST_FILE)
    for v in FEATURE_VARS:
        tr = load_var(v)
        clim = monthly_clim(tr)
        anom_tr = loo_anom(tr, clim)
        da_tr = xr.DataArray(anom_tr, dims=tr.dims, coords=tr.coords)
        # teste: anomalia vs clim do mes de ORIGEM (time_origem = alvo - 1)
        te = test[v]
        anom_te = te.values - clim[test.time_origem.dt.month.values - 1]
        da_te = xr.DataArray(anom_te, dims=te.dims, coords=te.coords)
        for rname, box in REGIONS.items():
            if f"{v}__{rname}" not in rows:
                rows[f"{v}__{rname}"] = []
            rows[f"{v}__{rname}"].append(region_mean(da_tr, box).values)
            rows[f"{v}__{rname}"].append(region_mean(da_te, box).values)
    times = np.concatenate([tr.time.values, test.time_origem.values])
    reg = pd.DataFrame({k: np.concatenate(v) for k, v in rows.items()},
                       index=pd.DatetimeIndex(times))
    reg = reg[~reg.index.duplicated(keep="first")]

    df = idx.join(reg, how="outer")
    df["sin_m"] = np.sin(2 * np.pi * df.index.month / 12)
    df["cos_m"] = np.cos(2 * np.pi * df.index.month / 12)
    # lags + medias moveis 3m dos indices (causal: mes m-k <= M-1)
    LAG_COLS = ["nino34", "amm", "atl3", "tna", "tsa", "npac", "ep_cp",
                "saod", "tio", "sam", "pdo"]
    for c in LAG_COLS:
        if c not in df:
            continue
        for k in (1, 2, 3):
            df[f"{c}_l{k}"] = df[c].shift(k)
        df[f"{c}_3m"] = df[c].rolling(3).mean()
    for col, wins in INTERACTIONS:
        for w in wins:
            mask = df.index.month.map(lambda m: SEASON_OF_MONTH[m] == w)
            df[f"{col}__{w}"] = df[col].where(mask, 0.0)
    df = df.loc["1940":"2024-11"].astype(np.float32)  # ultimo M-1 util = nov/2024
    df.to_parquet(EXT_DIR / "features_shared.parquet")
    return df


def production_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Congela o feature set de producao (D-016): descarta as cols adicionadas
    na expansao 124-col (SAM, PDO-EOF, NMME), que degradou o LB. Mantem
    'pdo_x_nino34' — interacao npac-based original da tabela 108-col."""
    def is_new(c: str) -> bool:
        return (c.startswith(("sam", "nmme"))
                or c == "pdo" or c.startswith("pdo_l") or c == "pdo_3m")
    return df.drop(columns=[c for c in df.columns if is_new(c)])


if __name__ == "__main__":
    df = build_shared_table()
    print(df.shape)
    print(df.iloc[:, :8].tail(8).round(2))
    print("NaN:", int(df.isna().sum().sum()))
