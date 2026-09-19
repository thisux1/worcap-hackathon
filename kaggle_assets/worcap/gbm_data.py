"""Matriz de features por pixel para GBM (V1).

Regime de precipitacao no teste (D-007): a ultima obs e' sempre dez/2022 e
lag_meses vai de 1 a 24. No treino o ultimo obs real e' sempre lag 1, entao
usamos LAG AUGMENTATION (D-012): para cada row de treino sorteia-se
lag ~ U{1..24} e a "ultima obs" vira a anomalia de precip do mes m-lag+1.

Memoria: vars ficam lazy no disco; guardamos apenas soma_por_mes (12,lat,lon).
Anomalia LOO da row i: (x*n - soma_m)/(n-1)  (D-013).
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter

from .config import DATA, FEATURE_FILES, FEATURE_VARS, TEST_FILE, TP_FILE, TARGET_FILE

SHARED_KEEP = None  # None = todas as colunas da tabela compartilhada
MAX_LAG = 24
ALL_VARS = FEATURE_VARS + ["tp"]

BASE_COLS = (
    [c for v in FEATURE_VARS for c in (f"{v}_a", f"{v}_n")]
    + ["tp_lastobs_a", "lag_meses", "clim_alvo", "sigma_alvo",
       "lat", "lon", "sin_tgt", "cos_tgt"]
)


def cols_for(shared) -> list:
    return BASE_COLS + list(SHARED_KEEP or shared.columns)


class Store:
    """Acesso as 9 vars + tp com anomalia LOO sob demanda.

    WORCAP_EAGER=1 carrega tudo em RAM (~3,1 GB) — usar onde ha memoria
    (Kaggle ~29 GB). Default: lazy, fatias mensais lidas do disco.
    """

    def __init__(self, eager: bool = None):
        import os
        if eager is None:
            eager = os.environ.get("WORCAP_EAGER") == "1"
        self.das, self.sums, self.cnts = {}, {}, {}
        for v in ALL_VARS:
            f = FEATURE_FILES.get(v, TP_FILE)
            da = xr.open_dataset(DATA / f)[v]
            if eager:
                da = da.load()
            g = da.groupby("time.month")
            self.sums[v] = g.sum("time").values.astype(np.float32)
            self.cnts[v] = np.array(
                [int((da.time.dt.month == m).sum()) for m in range(1, 13)],
                np.int32)
            self.das[v] = da
        self.t_idx = pd.DatetimeIndex(self.das["tp"].time.values)
        self.months = self.das["tp"].time.dt.month.values
        self.lat = self.das["tp"].lat.values.astype(np.float32)
        self.lon = self.das["tp"].lon.values.astype(np.float32)
        self.npix = self.das["tp"].sizes["lat"] * self.das["tp"].sizes["lon"]

    def raw(self, v, i):
        return self.das[v].isel(time=i).values.astype(np.float32)

    def anom(self, v, i):
        """Anomalia LOO vs os demais anos do mesmo mes-calendario."""
        x = self.raw(v, i)
        m = self.months[i]
        n = int(self.cnts[v][m - 1])
        return (x * n - self.sums[v][m - 1]) / (n - 1)

    def var_clim(self, v):
        return self.sums[v] / self.cnts[v][:, None, None]

    def close(self):
        for da in self.das.values():
            da.dataset.close()


def nbhd(a, size=5):
    return uniform_filter(a, size=size, mode="nearest")


def _pixel_block(store, i, tp_lo, lag, clim_t, sig_t, tgt_m,
                 shared_row, pix, nbhd_cache=None):
    n = len(pix)
    latg = np.repeat(store.lat, len(store.lon))
    long = np.tile(store.lon, len(store.lat))
    cols = []
    for v in FEATURE_VARS:
        a = store.anom(v, i)
        cols.append(a.ravel()[pix])
        cols.append(nbhd(a).ravel()[pix])
    cols += [
        tp_lo.ravel()[pix],
        np.full(n, lag, np.float32),
        clim_t.ravel()[pix],
        sig_t.ravel()[pix],
        latg[pix], long[pix],
        np.full(n, np.sin(2 * np.pi * tgt_m / 12), np.float32),
        np.full(n, np.cos(2 * np.pi * tgt_m / 12), np.float32),
    ]
    for k in (SHARED_KEEP or shared_row.index):
        cols.append(np.full(n, shared_row[k], np.float32))
    return np.stack(cols, 1)


def build_train_matrix(store, alvo_da, clim, sigma, shared, train_years,
                       n_pix=6000, seed=0, lag_aug=True):
    """X, z (anomalia padronizada do mes alvo), meta anos-alvo."""
    rng = np.random.default_rng(seed)
    t_idx, om = store.t_idx, store.months
    tgt_month = t_idx + pd.offsets.MonthBegin(1)
    tm, ty = tgt_month.month.values, tgt_month.year.values
    ok = set(train_years)
    rows = [i for i in range(len(t_idx)) if ty[i] in ok]
    c_all = clim.values
    s_all = np.maximum(sigma.values, 1e-3)

    Xs, zs, yrs = [], [], []
    for i in rows:
        y = alvo_da.isel(time=i).values.astype(np.float32)
        if np.isnan(y).all():
            continue
        lag = int(rng.integers(1, MAX_LAG + 1)) if lag_aug else 1
        obs_i = i - (lag - 1)
        if obs_i < 0:
            continue
        tp_lo = store.raw("tp", obs_i) - c_all[om[obs_i] - 1]
        pix = rng.choice(store.npix, n_pix, replace=False)
        Xs.append(_pixel_block(store, i, tp_lo, lag, c_all[tm[i] - 1],
                               s_all[tm[i] - 1], tm[i],
                               shared.loc[t_idx[i]], pix))
        zs.append(np.nan_to_num(((y - c_all[tm[i] - 1]) / s_all[tm[i] - 1])
                                .ravel()[pix]))
        yrs.append(np.full(n_pix, ty[i], np.int16))
    return (pd.DataFrame(np.concatenate(Xs), columns=cols_for(shared)),
            np.concatenate(zs), np.concatenate(yrs))


def _eval_block(store, i, obs_i, lag, clim, sigma, shared, tgt_i_month,
                tgt_i_time):
    om = store.months
    c_all, s_all = clim.values, np.maximum(sigma.values, 1e-3)
    tp_lo = store.raw("tp", obs_i) - c_all[om[obs_i] - 1]
    pix = np.arange(store.npix)
    X = _pixel_block(store, i, tp_lo, lag, c_all[tgt_i_month - 1],
                     s_all[tgt_i_month - 1], tgt_i_month,
                     shared.loc[store.t_idx[i]], pix)
    meta = {"time": tgt_i_time, "c": c_all[tgt_i_month - 1],
            "s": s_all[tgt_i_month - 1]}
    return X, meta


def build_eval_matrix(store, alvo_da, clim, sigma, shared, test_year,
                      anchor_gap=1):
    """Todos os pixels do test_year sob regime real de ancora.

    anchor_gap=1 -> ultima obs dez/(Y-1), lags 1..12 (tipo 2023).
    anchor_gap=2 -> ultima obs dez/(Y-2), lags 13..24 (tipo 2024).
    """
    t_idx = store.t_idx
    tgt_month = t_idx + pd.offsets.MonthBegin(1)
    ty, tm = tgt_month.year.values, tgt_month.month.values
    rows = [i for i in range(len(t_idx)) if ty[i] == test_year]
    anchor = int(np.where(t_idx == pd.Timestamp(f"{test_year-anchor_gap}-12-01"))[0][0])
    Xs, metas, ys = [], [], []
    for i in rows:
        lag = i - anchor + 1
        if lag < 1:
            continue
        X, meta = _eval_block(store, i, anchor, lag, clim, sigma, shared,
                              tm[i], tgt_month[i])
        Xs.append(X)
        metas.append(meta)
        ys.append(alvo_da.isel(time=i).values.astype(np.float32))
    return (pd.DataFrame(np.concatenate(Xs), columns=cols_for(shared)),
            metas, np.stack(ys))


def build_test_matrix(store, clim, sigma, shared):
    """X dos 24 alvos oficiais (todos os pixels) + metas p/ reconstruir."""
    test = xr.open_dataset(DATA / TEST_FILE)
    T = pd.DatetimeIndex(test.time.values)
    orig = pd.DatetimeIndex(test.time_origem.values)
    pix = np.arange(store.npix)
    c_all = clim.values
    s_all = np.maximum(sigma.values, 1e-3)
    tp_dec = c_all[11]
    Xs, metas = [], []
    for j, t in enumerate(T):
        om_ = orig[j].month
        cols = []
        n = store.npix
        latg = np.repeat(store.lat, len(store.lon))
        long = np.tile(store.lon, len(store.lat))
        for v in FEATURE_VARS:
            a = test[v].values[j].astype(np.float32) - store.var_clim(v)[om_ - 1]
            cols.append(a.ravel()[pix])
            cols.append(nbhd(a).ravel()[pix])
        tp_lo = test["tp_ultima_obs"].values[j].astype(np.float32) - tp_dec
        cols += [tp_lo.ravel()[pix],
                 np.full(n, int(test["lag_meses"].values[j]), np.float32),
                 c_all[t.month - 1].ravel()[pix],
                 s_all[t.month - 1].ravel()[pix],
                 latg, long,
                 np.full(n, np.sin(2 * np.pi * t.month / 12), np.float32),
                 np.full(n, np.cos(2 * np.pi * t.month / 12), np.float32)]
        sr = shared.loc[orig[j]]
        for k in (SHARED_KEEP or shared.columns):
            cols.append(np.full(n, sr[k], np.float32))
        Xs.append(np.stack(cols, 1))
        metas.append({"time": t, "c": c_all[t.month - 1],
                      "s": s_all[t.month - 1]})
    X = pd.DataFrame(np.concatenate(Xs), columns=cols_for(shared))
    return X, metas
