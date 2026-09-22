"""Climatologia condicional a fase ENSO — pos-processamento causal.

Ideia: o baseline usa climatologia neutra clim[m] (media de todos os anos de
treino). Aqui estimamos compositos por fase ENSO e corrigimos a previsao:

    y_hat += beta * (clim_fase[mes_alvo] - clim_full[mes_alvo])

Causalidade: a fase do alvo T e' decidida pelo nino34 do mes de ORIGEM
(T-1) — mesmo vintage das features compartilhadas (D-007), disponivel no
momento da previsao. Compositos sao fitados SOMENTE com os train_years do
fold (LOYO-safe). beta=0 reproduz o baseline (backward compat).
"""
import warnings

import numpy as np
import pandas as pd
import xarray as xr

WARM, COLD = 0.5, -0.5   # limiares ONI-like sobre nino34 (anomalia, degC)


def phase_of(nino34, warm=WARM, cold=COLD):
    """Fase ENSO: +1 El Nino (>= warm), -1 La Nina (<= cold), 0 neutro.

    Aceita escalar ou array; NaN -> 0 (neutro/conservador).
    """
    x = np.asarray(nino34, dtype=np.float64)
    out = np.zeros(x.shape, np.int8)
    out[np.isfinite(x) & (x >= warm)] = 1
    out[np.isfinite(x) & (x <= cold)] = -1
    return int(out) if out.ndim == 0 else out


def _n34_aligned(alvo_da: xr.DataArray, nino34_by_month) -> np.ndarray:
    """nino34 por mes de ORIGEM alinhado a alvo_da.time.

    Aceita pd.Series indexada por mes-origem (reindexa em alvo.time) ou
    array ja alinhado (mesmo comprimento que alvo.time).
    """
    if isinstance(nino34_by_month, pd.Series):
        return nino34_by_month.reindex(
            pd.DatetimeIndex(alvo_da.time.values)).values.astype(np.float64)
    arr = np.asarray(nino34_by_month, dtype=np.float64)
    assert arr.shape[0] == alvo_da.sizes["time"], \
        "nino34_by_month array deve estar alinhado a alvo.time (mes-origem)"
    return arr


def phase_composites(alvo_da: xr.DataArray, train_years, nino34_by_month,
                     warm=WARM, cold=COLD) -> dict:
    """{phase: ndarray (12, lat, lon)} — climatologia condicional por fase.

    alvo_da e' o alvo deslocado (treino_tp_alvo) indexado pelo mes de
    ORIGEM (alvo.time = mes M-1; ver load_shared em pixel_ridge).
    clim_phase[m] = media de alvo[i] sobre i com mes-ALVO m, ano-alvo em
    train_years e fase(nino34[mes-origem i]) == phase. Como o mes-origem
    do alvo m e' sempre o mes-calendario m-1, cada (m, phase) agrupa um
    unico mes por ano de treino. Celulas sem amostra ficam NaN.
    """
    t_idx = pd.DatetimeIndex(alvo_da.time.values)
    tgt = t_idx + pd.offsets.MonthBegin(1)
    ty, tm = tgt.year.values, tgt.month.values
    ph = phase_of(_n34_aligned(alvo_da, nino34_by_month), warm, cold)
    Y = alvo_da.values
    ok = np.isin(ty, list(train_years))
    out = {}
    for p in (1, 0, -1):
        comp = np.full((12,) + Y.shape[1:], np.nan, np.float32)
        for m in range(1, 13):
            sel = np.where(ok & (tm == m) & (ph == p))[0]
            if len(sel):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # nanmean de fatia NaN
                    comp[m - 1] = np.nanmean(Y[sel], axis=0)
        out[p] = comp
    return out


def enso_offset(phase_map: dict, full_clim, beta: float, phase: int):
    """Offset aditivo beta * (clim_fase - clim_full) para a fase dada.

    phase_map[phase] e full_clim: (12, lat, lon); retorna mesmo shape —
    o caller indexa [mes_alvo - 1] e ravela. phase == 0 (neutro) ou
    beta == 0 -> zeros. NaN no composito -> offset 0 (fallback p/ clim
    neutro naquele mes/pixel).
    """
    fc = np.asarray(full_clim, np.float32)
    if phase == 0 or beta == 0 or phase not in phase_map:
        return np.zeros(fc.shape, np.float32)
    delta = np.asarray(phase_map[phase], np.float32) - fc
    return (beta * np.nan_to_num(delta)).astype(np.float32)


# ---------------------------------------------------------------------------
# Compositos 2-way: (mes_alvo, fase_idx1, bin_idx2) — ex. ENSO x Atlantico.
# ---------------------------------------------------------------------------

def idx2_edges(idx2_vals, train_sel, bins2):
    """Cortes de bin do segundo indice (leakage-safe).

    bins2:
      "sign"            -> edges [0.0] (2 bins: neg / pos)
      "tercile"         -> quantis 1/3 e 2/3 de idx2 SOMENTE nos meses com
                         train_sel True (origens cujo ano-alvo e' de treino).
                         <3 valores finitos -> degenera p/ "sign".
      array-like        -> cortes explicitos (np.digitize).
    Retorna ndarray de edges; n_bins = len(edges) + 1.
    """
    if isinstance(bins2, str):
        if bins2 == "sign":
            return np.array([0.0])
        if bins2 in ("tercile", "terciles"):
            v = np.asarray(idx2_vals, np.float64)[train_sel]
            v = v[np.isfinite(v)]
            if len(v) < 3:
                return np.array([0.0])
            return np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
        raise ValueError(f"bins2 desconhecido: {bins2!r}")
    return np.asarray(bins2, np.float64)


def assign_bins(idx2_vals, edges):
    """Bin inteiro por valor (0..len(edges)); NaN -> -1 (fallback 1-way)."""
    x = np.asarray(idx2_vals, np.float64)
    b = np.digitize(x, np.asarray(edges, np.float64)).astype(np.int8)
    b[~np.isfinite(x)] = -1
    return b


def phase_composites_2way(alvo_da: xr.DataArray, train_years,
                          idx1_by_month, idx2_by_month,
                          thr1=0.8, bins2="sign", k=0.0) -> dict:
    """Compositos condicionais (mes_alvo, fase_idx1, bin_idx2).

    idx1_by_month: serie/array de nino34 no mes-origem (fase via +-thr1).
    idx2_by_month: segundo indice no mes-origem (tna, atl3, nino12, ep_cp),
        discretizado por `bins2` ("sign", "tercile" fitado no treino, ou
        edges explicitos). k: shrinkage do composito da celula para o
        composito 1-way da fase — cell' = n/(n+k)*cell + k/(n+k)*fase1.

    Fallbacks mensais (na ordem): celula vazia (n=0) -> composito
    (mes, fase1); pixel/linha todo-NaN -> idem; fase1 vazia -> NaN (o
    offset converte p/ 0 = clim_full). Causalidade identica a
    phase_composites: idx1/idx2 sao do mes-origem; edges de "tercile"
    usam so meses de treino (sem ano-alvo de eval).

    Retorna {"cells": {(fase,bin): (12,lat,lon)},
             "phase1": {fase: (12,lat,lon)},   # = phase_composites()
             "edges": ndarray, "n": {(fase,bin): (12,) int}}.
    """
    t_idx = pd.DatetimeIndex(alvo_da.time.values)
    tgt = t_idx + pd.offsets.MonthBegin(1)
    ty, tm = tgt.year.values, tgt.month.values
    ph = phase_of(_n34_aligned(alvo_da, idx1_by_month), thr1, -thr1)
    x2 = _n34_aligned(alvo_da, idx2_by_month)   # mesmo alinhamento
    Y = alvo_da.values
    ok = np.isin(ty, list(train_years))
    edges = idx2_edges(x2, ok, bins2)
    b2 = assign_bins(x2, edges)
    nbins = len(edges) + 1
    phase1 = phase_composites(alvo_da, train_years, idx1_by_month,
                              thr1, -thr1)
    cells, counts = {}, {}
    for p in (1, 0, -1):
        for bn in range(nbins):
            comp = np.full((12,) + Y.shape[1:], np.nan, np.float32)
            n_arr = np.zeros(12, np.int32)
            for m in range(1, 13):
                sel = np.where(ok & (tm == m) & (ph == p)
                               & (b2 == bn))[0]
                n = len(sel)
                n_arr[m - 1] = n
                fb = phase1[p][m - 1]
                if n == 0:
                    comp[m - 1] = fb          # fallback 1-way (pode ser NaN)
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cm = np.nanmean(Y[sel], axis=0)
                w = n / (n + k)               # k=0 -> w=1 (cell puro)
                shrunk = (w * np.nan_to_num(cm)
                          + (1.0 - w) * np.nan_to_num(fb))
                comp[m - 1] = np.where(np.isfinite(cm) | np.isfinite(fb),
                                       shrunk, np.nan)
            cells[(p, bn)] = comp
            counts[(p, bn)] = n_arr
    return {"cells": cells, "phase1": phase1, "edges": edges,
            "n": counts, "nbins": nbins}


def enso_offset_2way(phase_map2: dict, full_clim, beta: float, phase: int,
                     bin2):
    """Offset aditivo beta*(clim_2way - clim_full) para (fase, bin) dados.

    phase == 0 (neutro), beta == 0 -> zeros. bin2 None ou <0 (idx2 NaN) ou
    celula inexistente -> composito 1-way da fase (phase_map2["phase1"]);
    se tambem faltar -> zeros. NaN residual -> 0 (clim_full) por pixel.
    """
    fc = np.asarray(full_clim, np.float32)
    if phase == 0 or beta == 0 or not phase_map2:
        return np.zeros(fc.shape, np.float32)
    b = -1 if bin2 is None else (
        int(bin2) if np.isfinite(bin2) else -1)
    comp = phase_map2["cells"].get((phase, b)) if b >= 0 else None
    if comp is None:
        comp = phase_map2["phase1"].get(phase)
    if comp is None:
        return np.zeros(fc.shape, np.float32)
    delta = np.asarray(comp, np.float32) - fc
    return (beta * np.nan_to_num(delta)).astype(np.float32)
