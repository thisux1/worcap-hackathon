"""Teste sintetico de worcap.enso 2-way — sem dados reais.

Uso: .venv/bin/python tests/test_enso2way.py   (ou pytest tests/test_enso2way.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worcap.enso import (phase_of, phase_composites, enso_offset,
                         idx2_edges, assign_bins, phase_composites_2way,
                         enso_offset_2way)

rng = np.random.default_rng(0)
# origens jan/2000..dez/2003 -> alvos fev/2000..jan/2004
T = pd.date_range("2000-01-01", periods=48, freq="MS")
Y = rng.normal(size=(48, 3, 2)).astype(np.float32)
ALVO = xr.DataArray(Y, dims=("time", "lat", "lon"),
                    coords={"time": T, "lat": [0, 1, 2], "lon": [10, 20]})
# idx1 = nino34: quente se mes-origem em jan..jun, frio jul..dez
N34 = pd.Series(np.where(T.month <= 6, 1.0, -1.0), index=T)
# idx2: positivo em meses pares de origem, negativo nos impares
IDX2 = pd.Series(np.where(T.month % 2 == 0, 1.0, -1.0), index=T)


def _i(ts):
    return int(np.where(T == pd.Timestamp(ts))[0][0])


def test_edges_and_bins():
    tr = np.isfinite(N34.values)          # tudo "treino" aqui
    e = idx2_edges(IDX2.values, tr, "sign")
    np.testing.assert_array_equal(e, [0.0])
    e3 = idx2_edges(IDX2.values, tr, "tercile")
    assert len(e3) == 2                   # tercis -> 3 bins
    b = assign_bins(np.array([-2.0, 0.0, 5.0, np.nan]), [0.0])
    assert list(b) == [0, 1, 1, -1]       # NaN -> -1
    b3 = assign_bins(np.array([-1.0, 0.0, 1.0]), [-0.5, 0.5])
    assert list(b3) == [0, 1, 2]


def test_tercile_edges_train_only():
    """Quantis fitados SO em meses de treino — eval nao contamina."""
    x2 = np.zeros(len(T))
    tr = np.zeros(len(T), bool)
    tr[:24] = True                        # "treino" = 2 primeiros anos
    x2[24:] = 1e6                         # eval com outlier extremo
    e = idx2_edges(x2, tr, "tercile")
    np.testing.assert_allclose(e, np.quantile(np.zeros(24), [1/3, 2/3]))
    assert np.isfinite(e).all() and (e < 1e5).all()
    # degenerado: <3 valores finitos de treino -> cai p/ sign
    tr2 = np.zeros(len(T), bool); tr2[0] = True
    np.testing.assert_array_equal(idx2_edges(x2, tr2, "tercile"), [0.0])


def test_shapes_and_counts():
    pm2 = phase_composites_2way(ALVO, [2001, 2002], N34, IDX2,
                                thr1=0.5, bins2="sign", k=0.0)
    assert pm2["cells"][(1, 1)].shape == (12, 3, 2)
    assert pm2["phase1"][1].shape == (12, 3, 2)
    assert pm2["nbins"] == 2 and len(pm2["edges"]) == 1
    # por (mes, fase): n da soma dos bins == n do composito 1-way
    pm1 = phase_composites(ALVO, [2001, 2002], N34, 0.5, -0.5)
    for p in (1, 0, -1):
        tot = pm2["n"][(p, 0)] + pm2["n"][(p, 1)]
        for m in range(1, 13):
            tgt_m = T + pd.offsets.MonthBegin(1)
            exp = int((((tgt_m.year == 2001) | (tgt_m.year == 2002))
                       & (tgt_m.month == m)
                       & (phase_of(N34.values, .5, -.5) == p)).sum())
            assert tot[m - 1] == exp
            if exp == 0:
                assert np.isnan(pm1[p][m - 1]).all()


def test_cell_values_and_fallback():
    pm2 = phase_composites_2way(ALVO, [2001], N34, IDX2,
                                thr1=0.5, bins2="sign", k=0.0)
    # alvo abr/2001 (m=4): origem mar/2001 -> quente, idx2 mar(impar)=neg
    i = _i("2001-03-01")
    np.testing.assert_allclose(pm2["cells"][(1, 0)][3], Y[i])
    # mesma celula para alvo mai/2001: origem abr(par)=pos -> celula (1,0)
    # vazia -> fallback exato p/ composito 1-way (origem abr/2001)
    np.testing.assert_allclose(pm2["cells"][(1, 0)][4],
                               pm2["phase1"][1][4])
    np.testing.assert_allclose(pm2["cells"][(1, 0)][4], Y[_i("2001-04-01")])
    # mes fora do alcance da fase -> NaN nos dois niveis
    assert np.isnan(pm2["cells"][(1, 0)][0]).all()   # jan/01: origem dez/00 frio
    # consistencia: celula cheia (k=0) == nanmean do subconjunto certo
    pm1 = phase_composites(ALVO, [2001], N34, 0.5, -0.5)
    np.testing.assert_allclose(pm2["phase1"][1], pm1[1])


def test_shrinkage_math():
    """n=1, k=1 -> cell' = 0.5*cell + 0.5*fase1 no mes-alvo."""
    pm2 = phase_composites_2way(ALVO, [2001, 2002], N34, IDX2,
                                thr1=0.5, bins2="sign", k=1.0)
    # alvo abr (m=4): origens mar/01 e mar/02 quentes; idx2 mar = neg (bin0)
    i1, i2 = _i("2001-03-01"), _i("2002-03-01")
    cell = 0.5 * (Y[i1] + Y[i2])
    np.testing.assert_allclose(pm2["phase1"][1][3], cell, rtol=1e-6)
    np.testing.assert_allclose(pm2["n"][(1, 0)][3], 2)
    np.testing.assert_allclose(pm2["n"][(1, 1)][3], 0)  # abr caiu p/ fallback
    # celula (1,0) com n=2, k=1 -> w=2/3: 2/3*cell + 1/3*fase1 = cell aqui
    np.testing.assert_allclose(pm2["cells"][(1, 0)][3], cell, rtol=1e-6)
    # alvo mai (m=5): origem abr quente, idx2 abr=pos -> (1,1) tem n=2;
    # (1,0) vazia -> fallback p/ fase1 mesmo com k>0
    np.testing.assert_allclose(pm2["cells"][(1, 0)][4],
                               pm2["phase1"][1][4])
    # caso distinguivel: idx2 so positivo em mar/01 -> celula n=1 vs fase n=2
    idx2b = pd.Series(np.where(T == pd.Timestamp("2001-03-01"), 1.0, -1.0),
                      index=T)
    pm3 = phase_composites_2way(ALVO, [2001, 2002], N34, idx2b,
                                thr1=0.5, bins2="sign", k=1.0)
    exp = 0.5 * Y[i1] + 0.5 * cell        # w = 1/(1+1)
    np.testing.assert_allclose(pm3["cells"][(1, 1)][3], exp, rtol=1e-6)


def test_offset_2way():
    pm2 = phase_composites_2way(ALVO, [2001], N34, IDX2,
                                thr1=0.5, bins2="sign", k=0.0)
    full = np.zeros((12, 3, 2), np.float32)
    i = _i("2001-03-01")
    off = enso_offset_2way(pm2, full, beta=0.5, phase=1, bin2=0)
    np.testing.assert_allclose(off[3], 0.5 * Y[i])
    # neutro -> 0 ; beta=0 -> 0
    assert enso_offset_2way(pm2, full, 0.5, 0, 0).sum() == 0
    assert enso_offset_2way(pm2, full, 0.0, 1, 0).sum() == 0
    # idx2 faltante (None / -1 / NaN) -> fallback 1-way = enso_offset classico
    ref = enso_offset(pm2["phase1"], full, 0.5, 1)
    for bn in (None, -1, np.nan):
        np.testing.assert_allclose(
            enso_offset_2way(pm2, full, 0.5, 1, bn), ref)
    # mes sem fase quente -> offset 0 (fallback final = clim_full)
    assert off[0].sum() == 0
    # bin inexistente -> fallback 1-way
    np.testing.assert_allclose(
        enso_offset_2way(pm2, full, 0.5, 1, 9), ref)


def test_explicit_edges():
    pm2 = phase_composites_2way(ALVO, [2001], N34, IDX2,
                                thr1=0.5, bins2=[-0.5, 0.5], k=0.0)
    assert pm2["nbins"] == 3              # 2 edges -> 3 bins
    # idx2 = +-1 -> nunca cai no bin central
    assert pm2["n"][(1, 1)].sum() == 0
    np.testing.assert_allclose(np.nan_to_num(pm2["cells"][(1, 1)]),
                               np.nan_to_num(pm2["phase1"][1]))


if __name__ == "__main__":
    for f in list(vars().values()):
        if callable(f) and getattr(f, "__name__", "").startswith("test_"):
            f()
    print("test_enso2way: OK")
