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
