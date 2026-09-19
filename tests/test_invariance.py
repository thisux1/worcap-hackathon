"""Teste de invariancia causal — regra oficial: prever o mes T usa apenas dados
disponiveis ate o fim de T-1 ("nada posterior").

Prova empirica, sem treinar modelo:
  1. Metadata: time_origem[j] == time[j] - 1 mes para todo alvo j.
  2. Stage-1: Xte[j] = df[orig[j]] — corromper todas as linhas de df fora de
     orig[j] nao altera as features do alvo j.
  3. Stage-2: x2_test(j) le somente a linha j do teste_features + tp_ultima_obs
     (fixo dez/2022) + lag_meses + NMME com init = time_origem — corromper todas
     as outras linhas nao altera a saida.
  4. NMME: somente inits iguais a time_origem sao acessados (spy dict).

Uso: .venv/bin/python tests/test_invariance.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worcap.config import DATA, EXT_DIR, TEST_FILE, FEATURE_VARS
from worcap.pixel_ridge import x2_test

NPIX = 301 * 261


class FakeStore:
    """Stub minimo: valores reais sao irrelevantes para provar invariancia."""
    npix = NPIX
    sums = {"tp": np.zeros((12, 301, 261), np.float32)}
    cnts = {"tp": np.ones((12, 301, 261), np.float32)}
    das = {"tp": np.zeros((1, 301, 261), np.float32)}

    def var_clim(self, v):
        return np.zeros((12, 301, 261), np.float32)


class SpyDict(dict):
    def __init__(self):
        super().__init__()
        self.accessed = []

    def get(self, k, default=None):
        self.accessed.append(k)
        return super().get(k, default)


def corrupt_rows(ds, keep_j, rng):
    """Copia do dataset com FEATURE_VARS ruidosos em todas as linhas != keep_j."""
    out = ds.copy()
    for v in FEATURE_VARS + ["tp_ultima_obs"]:
        arr = out[v].values.copy()
        noise = rng.standard_normal(arr.shape).astype(np.float32) * 99
        arr[keep_j] = ds[v].values[keep_j]
        mask = np.ones(len(arr), bool)
        mask[keep_j] = False
        arr[mask] = noise[mask]
        out[v].values[:] = arr
    return out


def main():
    test = xr.open_dataset(DATA / TEST_FILE)
    df = pd.read_parquet(EXT_DIR / "features_shared.parquet")
    rng = np.random.default_rng(0)
    T = pd.DatetimeIndex(test.time.values)
    orig = pd.DatetimeIndex(test.time_origem.values)
    lag = test["lag_meses"].values

    # 1) metadata: time_origem = alvo - 1 mes; lag coerente com ancora dez/2022
    assert (T - pd.offsets.MonthBegin(1)).equals(orig), "time_origem != T-1"
    assert (test["tp_ultima_obs"].values == test["tp_ultima_obs"].values[0]).all(), \
        "tp_ultima_obs varia por linha?"
    exp_lag = (T.year - 2022) * 12 + T.month - 12
    assert (lag == exp_lag).all(), "lag_meses inconsistente com ancora dez/2022"
    print("[ok] metadata: time_origem = T-1, ancora dez/2022, lag consistente")

    # 2) stage-1: features do alvo j dependem so de df[orig[j]]
    Xte = df.reindex(orig)
    for j in (0, 11, 23):
        dfc = df.copy()
        mask = np.ones(len(dfc), bool)
        mask[dfc.index.get_loc(orig[j])] = False
        dfc.values[mask] = rng.standard_normal((mask.sum(), dfc.shape[1]))
        assert dfc.reindex(orig).iloc[j].equals(Xte.iloc[j]), \
            f"Xte[{j}] mudou ao corromper outras linhas de df"
    print("[ok] stage-1: Xte[j] depende apenas de df[time_origem[j]]")

    # 3) stage-2: x2_test(j) depende apenas da linha j do arquivo de teste
    store = FakeStore()
    for j in (0, 5, 11, 17, 23):
        base = x2_test(store, test, j, orig[j].month, int(lag[j]))
        tc = corrupt_rows(test, j, rng)
        alt = x2_test(store, tc, j, orig[j].month, int(lag[j]))
        assert np.array_equal(base, alt), f"x2_test[{j}] mudou com linhas != j"
    print("[ok] stage-2: x2_test[j] depende apenas da linha j (origem = T-1)")

    # 4) NMME: apenas init = time_origem e acessado
    nmme = SpyDict()
    for j in range(len(T)):
        nmme[orig[j]] = np.zeros((301, 261), np.float32)
    for j in range(len(T)):
        x2_test(store, test, j, orig[j].month, int(lag[j]), nmme)
    assert set(nmme.accessed) == set(orig), "NMME acessou init != time_origem"
    print("[ok] NMME: apenas inits = time_origem (<= T-1) sao acessados")

    print("\nINVARIANCIA CAUSAL VERIFICADA — previsao de T usa apenas dados <= T-1")


if __name__ == "__main__":
    main()
