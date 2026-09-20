"""V2: ridge por pixel no residuo do ridge compartilhado (mini-Rodeo).

Stage 1: ridge por pixel sobre features mensais compartilhadas (v05).
Stage 2: por pixel, ridge sobre o RESIDUO com features que variam por pixel:
    intercept + 9 anomalias atmosfericas LOO + 9 vizinhanca suavizada
    + tp_lastobs_anom + lag + lag*tp_lastobs_anom          (k=22)

Custo: acumula Gram (npix,k,k) mes a mes — ~150 MB — sem matriz (n,npix,k).
Lag augmentation igual ao GBM (D-012): em cada mes de treino sorteia-se
lag ~ U{1..24}; anomalias atmosfericas sao do mes origem i, tp_lastobs de
i-(lag-1) (D-007: no teste so tp_lastobs fica velho, atm continua M-1).
"""
import numpy as np
import pandas as pd
import xarray as xr

from .config import DATA, TP_FILE, TARGET_FILE, TEST_FILE, FEATURE_VARS, EXT_DIR
from .climatology import fit_climatology
from .gbm_data import Store, nbhd, MAX_LAG

K2 = 2 + 2 * len(FEATURE_VARS) + 2   # bias + 9 a + 9 n + tp_lo + lag + lag*tp
K2_NMME = K2 + 1                     # + anomalia NMME debiased (init do mes)


def load_shared():
    alvo = xr.open_dataset(DATA / TARGET_FILE)["tp_alvo"]
    df = pd.read_parquet(EXT_DIR / "features_shared.parquet")
    X = df.reindex(pd.DatetimeIndex(alvo.time.values)).fillna(0.0)
    tgt_month = pd.DatetimeIndex(alvo.time.values) + pd.offsets.MonthBegin(1)
    return alvo, X.values.astype(np.float64), tgt_month


def shared_ridge(Xraw, Ztr, rows_tr, lam, winsorize_sigma=None):
    """Ridge compartilhado sobre features padronizadas (mu/sd de rows_tr).

    winsorize_sigma: se nao None (ex. 4.0), clipa Xn em [-w, +w] — limita a
    influencia de features fora-de-distribuicao no ano de eval/teste (ex. SAM
    +4.28 sigma em dez/2023). Coluna 0 (intercept) nunca e clipada.
    """
    p = np.eye(Xraw.shape[1]) * lam
    p[0, 0] = 0.0
    mu = Xraw[rows_tr].mean(0)
    sd = Xraw[rows_tr].std(0); sd[sd == 0] = 1
    mu[0], sd[0] = 0.0, 1.0
    Xn = (Xraw - mu) / sd
    if winsorize_sigma is not None:
        Xn[:, 1:] = np.clip(Xn[:, 1:], -winsorize_sigma, winsorize_sigma)
    W = np.linalg.solve(Xn[rows_tr].T @ Xn[rows_tr] + p,
                        Xn[rows_tr].T @ Ztr)
    return Xn, W, mu, sd


def x2_month(store, i, obs_i, lag, nmme=None, tp_lo=None, x2_stats=None):
    """Features per-pixel do mes origem i: (npix, K2|K2_NMME).

    nmme: dict {Timestamp(init): array(lat,lon) anomalia debiased} — init do
    mes de origem i (preve i+1). Meses sem init NMME -> zeros.
    tp_lo: override da anomalia de precip (flat npix) — usado na recursao
    (previsao do mes anterior vira "ultima obs"); obs_i ignorado nesse caso.
    x2_stats: (mu, sd) do model["x2_mu"/"x2_sd"] — padroniza todas as
    colunas exceto intercept (col 0, mu=0/sd=1) e lag fica padronizado
    tambem (col passa a ter variancia unitaria como as demais).
    """
    cols = [np.ones(store.npix, np.float32)]
    for v in FEATURE_VARS:
        a = store.anom(v, i)
        cols += [a.ravel(), nbhd(a).ravel()]
    if tp_lo is None:
        tp_lo = (store.raw("tp", obs_i)
                 - store.sums["tp"][store.months[obs_i] - 1]
                 / store.cnts["tp"][store.months[obs_i] - 1]).ravel()
    cols += [tp_lo, np.full(store.npix, lag, np.float32), lag * tp_lo]
    if nmme is not None:
        f = nmme.get(pd.Timestamp(store.t_idx[i]))
        cols.append(np.zeros(store.npix, np.float32) if f is None
                    else f.ravel().astype(np.float32))
    X2 = np.stack(cols, 1)
    if x2_stats is not None:
        X2 = (X2 - x2_stats[0]) / x2_stats[1]
    return X2


def _x2_stats(model):
    """(x2_mu, x2_sd) ou None — padronizacao de stage-2 desligada."""
    mu = model.get("x2_mu")
    return None if mu is None else (mu, model["x2_sd"])


def _nmme_map(train_years):
    """{Timestamp(init): campo (lat,lon) anomalia NMME} LOYO-safe.

    Clim do modelo usa so inits <=2016 fora dos anos do fold/embargo —
    aqui excluimos TODOS os anos fora de train_years (conservador).
    """
    from .nmme import load_raw, anom_on_grid
    raw = load_raw()
    t = pd.DatetimeIndex(raw.init.values)
    excl = [y for y in range(t.year.min(), 2017) if y not in train_years]
    an = anom_on_grid(raw, exclude_years=excl)
    return {t[i]: np.nan_to_num(an.isel(init=i).values.astype(np.float32))
            for i in range(len(t))}


def _grid_shape(store):
    """(n_lat, n_lon) da grade oficial."""
    return store.das["tp"].shape[1:]


def smooth_w2(store, W2, size=5):
    """Media por vizinhanca de cada mapa de coeficiente (npix,k).

    Mesmo kernel do feature `_n` (gbm_data.nbhd): uniform_filter size=5,
    mode="nearest" — consistencia com o smooth usado nas features.
    """
    nlat, nlon = _grid_shape(store)
    return np.stack([nbhd(W2[:, k].reshape(nlat, nlon), size=size).ravel()
                     for k in range(W2.shape[1])], axis=1)


def region_mean_w2(store, W2):
    """Media de W2 dentro das macro-regioes de features.REGIONS (npix,k).

    Pixel em >1 caixa recebe a media das medias regionais; pixel fora de
    todas recebe a media global da coluna.
    """
    from .features import REGIONS
    latg = np.repeat(store.lat, len(store.lon))
    long = np.tile(store.lon, len(store.lat))
    acc = np.zeros_like(W2)
    cnt = np.zeros(store.npix, np.int32)
    for latS, latN, lonW, lonE in REGIONS.values():
        m = (latg >= latS) & (latg <= latN) & (long >= lonW) & (long <= lonE)
        if m.any():
            acc[m] += W2[m].mean(0)
            cnt[m] += 1
    out = np.broadcast_to(W2.mean(0), W2.shape).copy()
    ok = cnt > 0
    out[ok] = acc[ok] / cnt[ok, None]
    return out


def pool_w2(store, W2, alpha=None, regional=None, intercept=True, size=5):
    """Partial pooling dos coeficientes stage-2 (Max-and-Smooth).

    alpha in [0,1]:  W2' = alpha*W2 + (1-alpha)*vizinhanca(W2)
        (0 = mapa todo suavizado, 1 = identidade).
    regional in [0,1]: shrink adicional para a media da macro-regiao,
        aplicado sobre o resultado do pooling espacial; o alvo regional
        e' computado sobre o W2 cru:  W2'' = r*W2' + (1-r)*R(W2).
    intercept=True encolhe tambem a coluna de bias (k=0). Justificativa:
        o intercepto e' o unico coeficiente NAO penalizado no ridge
        (P[0,0]=0) -> e' o mais ruidoso e o vies residual e' espacialmente
        coerente; Max-and-Smooth canonico suaviza todos os mapas.
        intercept=False preserva W2[:,0] (ablation/diagnostico).
    """
    out = W2
    if alpha is not None and alpha < 1:
        out = alpha * W2 + (1 - alpha) * smooth_w2(store, W2, size)
    if regional is not None:
        out = regional * out + (1 - regional) * region_mean_w2(store, W2)
    if not intercept and out is not W2:
        out = out.copy()
        out[:, 0] = W2[:, 0]
    return out


def fit_pixel_ridge(store, alvo, Xraw, tgt_month, train_years, lam1, lam2,
                    seed=0, lag_aug=True, use_nmme=False,
                    winsorize_sigma=None, x2_std=None,
                    pool_alpha=None, pool_regional=None, pool_intercept=True):
    """Stage1 (ridge compartilhado) + stage2 (ridge por pixel no residuo).

    winsorize_sigma: clipa Xn (stage-1) em +-w sigma — ver shared_ridge.
    x2_std: se True, padroniza as colunas de X2 (media/sd global por coluna,
    acumulados nas mesmas rows do fit; col 0 = intercept fica crua). Assim
    lam2 penaliza todas as features igualmente — sem isso features em
    unidades grandes (surface_pressure ~100 Pa, geopotential ~1e3) sao
    quase livres e features pequenas (shum_850 ~4e-3) morrem. A Gram
    padronizada e obtida em forma fechada a partir de G,b crus (identico a
    padronizar X2 antes de acumular, ate erro de arredondamento).
    pool_alpha/pool_regional (None = off): partial pooling espacial de W2
    via pool_w2 — o W2 cru e' preservado em model["W2_raw"].
    """
    rng = np.random.default_rng(seed)
    t_idx = store.t_idx
    ty, tm = tgt_month.year.values, tgt_month.month.values
    clim, sigma = fit_climatology(
        xr.open_dataset(DATA / TP_FILE)["tp"], train_years)
    c, s = clim.values, np.maximum(sigma.values, 1e-3)
    rows = np.array([i for i in range(len(t_idx)) if ty[i] in train_years])
    Y = alvo.values
    Ztr = np.nan_to_num(((Y[rows] - c[tm[rows] - 1]) / s[tm[rows] - 1])
                        .reshape(len(rows), -1))
    Xn, W, mu, sd = shared_ridge(Xraw, Ztr, rows, lam1,
                                 winsorize_sigma=winsorize_sigma)
    nmme = _nmme_map(train_years) if use_nmme else None
    k2 = K2_NMME if use_nmme else K2
    G = np.zeros((store.npix, k2, k2), np.float64)
    b = np.zeros((store.npix, k2), np.float64)
    Sx2 = np.zeros((store.npix, k2), np.float64)   # soma de X2 por pixel
    Rsum = np.zeros(store.npix, np.float64)        # soma do residuo por pixel
    n2 = 0                                         # rows que entraram no fit
    for i in rows:
        y = Y[i].ravel()
        if np.isnan(y).all():
            continue
        lag = int(rng.integers(1, MAX_LAG + 1)) if lag_aug else 1
        obs_i = i - (lag - 1)
        if obs_i < 0:
            continue
        z1 = (Xn[i] @ W).ravel()
        r = y - (c[tm[i] - 1].ravel() + s[tm[i] - 1].ravel() * z1)
        r = np.nan_to_num(r)
        X2 = x2_month(store, i, obs_i, lag, nmme)
        G += np.einsum("pk,pl->pkl", X2, X2)
        b += np.einsum("pk,p->pk", X2, r)
        if x2_std:
            Sx2 += X2
            Rsum += r
            n2 += 1
    model = {"Xn": Xn, "W": W, "mu": mu, "sd": sd,
             "clim": clim, "sigma": sigma, "nmme": nmme,
             "winsorize": winsorize_sigma}
    if x2_std and n2 > 0:
        # Media/sd globais por coluna sobre as n2* npix observacoes.
        # X2s = (X2 - x2_mu)/x2_sd = X2*D - c  (D=diag(d), c=mu*d) ->
        #   Gs = D G D - c (Sx2 D)' - (Sx2 D) c' + n2 c c'
        #   bs = b D - Rsum c            (por pixel)
        nt = n2 * store.npix
        x2_mu = Sx2.sum(0) / nt
        x2_var = np.einsum("pkk->k", G) / nt - x2_mu ** 2
        x2_sd = np.sqrt(np.maximum(x2_var, 0.0))
        x2_sd[x2_sd == 0] = 1.0
        x2_mu[0], x2_sd[0] = 0.0, 1.0            # intercept cru
        d = 1.0 / x2_sd
        cc = x2_mu * d
        G *= d[:, None] * d[None, :]
        G -= Sx2[:, None, :] * (cc[:, None] * d[None, :])
        G -= Sx2[:, :, None] * (d[:, None] * cc[None, :])
        G += n2 * (cc[:, None] * cc[None, :])
        b = b * d[None, :] - Rsum[:, None] * cc[None, :]
        model["x2_mu"], model["x2_sd"] = x2_mu, x2_sd
    P = np.eye(k2) * lam2
    P[0, 0] = 0.0
    W2 = np.linalg.solve(G + P, b[..., None])[..., 0]
    if pool_alpha is not None or pool_regional is not None:
        model["W2_raw"] = W2
        W2 = pool_w2(store, W2, pool_alpha, pool_regional,
                     pool_intercept)
    model["W2"] = W2
    return model


def x2_test(store, test_ds, j, om_, lag, nmme=None, x2_stats=None):
    """Stage2 features do alvo j do arquivo de teste (init NMME = time_origem).

    x2_stats: (mu, sd) do modelo — mesma padronizacao do fit (x2_std)."""
    cols = [np.ones(store.npix, np.float32)]
    vc = {v: store.var_clim(v) for v in FEATURE_VARS}
    for v in FEATURE_VARS:
        a = (test_ds[v].values[j].astype(np.float32) - vc[v][om_ - 1])
        cols += [a.ravel(), nbhd(a).ravel()]
    tp_lo = (test_ds["tp_ultima_obs"].values[j].astype(np.float32)
             - store.sums["tp"][11] / store.cnts["tp"][11]).ravel()
    cols += [tp_lo, np.full(store.npix, lag, np.float32), lag * tp_lo]
    if nmme is not None:
        f = nmme.get(pd.Timestamp(test_ds.time_origem.values[j]))
        cols.append(np.zeros(store.npix, np.float32) if f is None
                    else f.ravel().astype(np.float32))
    X2 = np.stack(cols, 1)
    if x2_stats is not None:
        X2 = (X2 - x2_stats[0]) / x2_stats[1]
    return X2


def eval_year(store, alvo, model, test_year, anchor_gap=1):
    """RMSE do ano sob regime de ancora real (1=tipo2023, 2=tipo2024).

    Retorna (rmse_stage1, rmse_stage1+2)."""
    t_idx = store.t_idx
    tgt_month = t_idx + pd.offsets.MonthBegin(1)
    ty, tm = tgt_month.year.values, tgt_month.month.values
    c = model["clim"].values
    s = np.maximum(model["sigma"].values, 1e-3)
    Xn, W, W2 = model["Xn"], model["W"], model["W2"]
    anchor = int(np.where(
        t_idx == pd.Timestamp(f"{test_year-anchor_gap}-12-01"))[0][0])
    Y = alvo.values
    e1 = e2 = n = 0.0
    for i in [i for i in range(len(t_idx)) if ty[i] == test_year]:
        lag = i - anchor + 1
        if lag < 1:
            continue
        y = Y[i].ravel()
        z1 = (Xn[i] @ W).ravel()
        base = c[tm[i] - 1].ravel() + s[tm[i] - 1].ravel() * z1
        X2 = x2_month(store, i, anchor, lag, model.get("nmme"),
                      x2_stats=_x2_stats(model))
        yh = base + np.einsum("pk,pk->p", X2, W2)
        e1 += np.nansum((np.clip(base, 0, None) - y) ** 2)
        e2 += np.nansum((np.clip(yh, 0, None) - y) ** 2)
        n += np.isfinite(y).sum()
    return np.sqrt(e1 / n), np.sqrt(e2 / n)


def eval_year_preds(store, alvo, model, test_year, anchor_gap=1, n34=None):
    """Previsoes do ano por mes sob regime de ancora (base s1 nao-clipada,
    corr s2, y, clim). Permite varrer peso de blend e damping offline.

    Cada dict tambem leva 'i' (idx do mes-origem em store.t_idx), 'tm'
    (mes-calendario alvo) e 'n34' (nino34 no mes-origem) — insumos do
    pos-processamento ENSO (worcap.enso). n34 pode vir do parametro
    (array/Series alinhado a store.t_idx) ou de model["n34"]; sem nenhum
    dos dois fica NaN. Default beta=0 no sweep => baseline inalterado.
    """
    t_idx = store.t_idx
    tgt_month = t_idx + pd.offsets.MonthBegin(1)
    ty, tm = tgt_month.year.values, tgt_month.month.values
    c = model["clim"].values
    s = np.maximum(model["sigma"].values, 1e-3)
    Xn, W, W2 = model["Xn"], model["W"], model["W2"]
    if n34 is None:
        n34 = model.get("n34")
    if isinstance(n34, pd.Series):
        n34 = n34.reindex(t_idx).values
    anchor = int(np.where(
        t_idx == pd.Timestamp(f"{test_year-anchor_gap}-12-01"))[0][0])
    Y = alvo.values
    out = []
    for i in [i for i in range(len(t_idx)) if ty[i] == test_year]:
        lag = i - anchor + 1
        if lag < 1:
            continue
        z1 = (Xn[i] @ W).ravel()
        base = c[tm[i] - 1].ravel() + s[tm[i] - 1].ravel() * z1
        X2 = x2_month(store, i, anchor, lag, model.get("nmme"),
                      x2_stats=_x2_stats(model))
        corr = np.einsum("pk,pk->p", X2, W2)
        out.append({"base": base, "corr": corr, "y": Y[i].ravel(),
                    "clim": c[tm[i] - 1].ravel(), "i": i, "tm": int(tm[i]),
                    "n34": float(n34[i]) if n34 is not None else np.nan})
    return out


def eval_year_rec(store, alvo, model, test_year):
    """Regime 2024 recursivo (variante R2): alvos de test_year-1 usam a
    ancora real dez(test_year-2); alvos de test_year usam tp_lo = previsao
    do mes anterior com lag=1. Score so nos meses de test_year.
    Retorna (rmse, rmse_por_posicao_na_cadeia)."""
    t_idx = store.t_idx
    tgt_month = t_idx + pd.offsets.MonthBegin(1)
    ty, tm = tgt_month.year.values, tgt_month.month.values
    c = model["clim"].values
    s = np.maximum(model["sigma"].values, 1e-3)
    Xn, W, W2 = model["Xn"], model["W"], model["W2"]
    anchor = int(np.where(
        t_idx == pd.Timestamp(f"{test_year-2}-12-01"))[0][0])
    clim_full = (store.sums["tp"]
                 / np.maximum(store.cnts["tp"], 1)[:, None, None])
    Y = alvo.values
    tp_prev = store.raw("tp", anchor).ravel()      # obs real dez(Y-2)
    e2 = n = 0.0
    per_step = []
    for k in range(24):
        i = anchor + k                              # mes-origem
        if i >= len(t_idx):
            break
        year_i = ty[i]
        if year_i < test_year - 1 or year_i > test_year:
            continue
        z1 = (Xn[i] @ W).ravel()
        base = c[tm[i] - 1].ravel() + s[tm[i] - 1].ravel() * z1
        if year_i == test_year - 1:
            X2 = x2_month(store, i, anchor, k + 1, model.get("nmme"),
                          x2_stats=_x2_stats(model))
        else:
            tp_lo = tp_prev - clim_full[store.months[i] - 1].ravel()
            X2 = x2_month(store, i, None, 1, model.get("nmme"),
                          tp_lo=tp_lo, x2_stats=_x2_stats(model))
        yh = np.clip(base + np.einsum("pk,pk->p", X2, W2), 0, None)
        tp_prev = yh
        if year_i == test_year:
            y = Y[i].ravel()
            e = np.nansum((yh - y) ** 2)
            nn = np.isfinite(y).sum()
            e2 += e
            n += nn
            per_step.append(np.sqrt(e / nn))
    return np.sqrt(e2 / n), per_step


def submit(store, alvo, Xraw, tgt_month, df, lam1, lam2, tag,
           all_years=True, seed=0, use_nmme=False,
           winsorize_sigma=None, x2_std=None, pool_alpha=None,
           enso_beta=None, enso_thr=0.8):
    """Fit em todos os anos (ou final_fit_years) + CSV dos 24 alvos."""
    from .folds import final_fit_years
    from .config import TRAIN_START, TRAIN_END, SUB_DIR
    from .submission import write_submission
    years = (list(range(TRAIN_START, TRAIN_END + 1)) if all_years
             else final_fit_years())
    model = fit_pixel_ridge(store, alvo, Xraw, tgt_month, years,
                            lam1, lam2, seed=seed, use_nmme=use_nmme,
                            winsorize_sigma=winsorize_sigma, x2_std=x2_std,
                            pool_alpha=pool_alpha)
    phase_map, n34 = None, None
    if enso_beta:
        from .enso import phase_composites, phase_of, enso_offset
        if "nino34" in df.columns:
            n34 = df["nino34"]
            phase_map = phase_composites(alvo, years, n34,
                                         enso_thr, -enso_thr)
    test = xr.open_dataset(DATA / TEST_FILE)
    T = pd.DatetimeIndex(test.time.values)
    orig = pd.DatetimeIndex(test.time_origem.values)
    Xte = df.reindex(orig).fillna(0.0).values.astype(np.float64)
    Xte = np.hstack([np.ones((len(Xte), 1)), Xte])
    Xte_n = (Xte - model["mu"]) / model["sd"]
    w = model.get("winsorize")
    if w is not None:
        Xte_n[:, 1:] = np.clip(Xte_n[:, 1:], -w, w)
    c = model["clim"].values
    s = np.maximum(model["sigma"].values, 1e-3)
    shape = store.das["tp"].shape[1:]
    ys = []
    for j, t in enumerate(T):
        lag = int(test["lag_meses"].values[j])
        z1 = Xte_n[j] @ model["W"]
        X2 = x2_test(store, test, j, orig[j].month, lag, model.get("nmme"),
                     x2_stats=_x2_stats(model))
        off = 0.0
        if phase_map is not None:
            ph = phase_of(n34.get(pd.Timestamp(orig[j]), np.nan))
            off = enso_offset(phase_map, c, enso_beta,
                              ph)[t.month - 1].ravel()
        yh = (c[t.month - 1].ravel() + s[t.month - 1].ravel() * z1
              + np.einsum("pk,pk->p", X2, model["W2"]) + off)
        ys.append(np.clip(yh.reshape(shape), 0, None))
    da = xr.DataArray(np.stack(ys), dims=("time", "lat", "lon"),
                      coords={"time": T, "lat": alvo.lat, "lon": alvo.lon})
    path = SUB_DIR / f"{tag}.csv"
    write_submission(da, path)
    return path
