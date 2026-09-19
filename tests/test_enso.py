"""Teste sintetico de worcap.enso — sem dados reais.

Uso: .venv/bin/python tests/test_enso.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worcap.enso import phase_of, phase_composites, enso_offset

# --- phase_of escalar ---
assert phase_of(1.0) == 1 and phase_of(0.5) == 1
assert phase_of(-0.7) == -1 and phase_of(-0.5) == -1
assert phase_of(0.49) == 0 and phase_of(-0.49) == 0 and phase_of(np.nan) == 0
assert phase_of(0.3, warm=0.2) == 1          # limiar custom
arr = phase_of(np.array([1.0, 0.0, -1.0, np.nan]))
assert list(arr) == [1, 0, -1, 0]

# --- phase_composites: alvo indexado pelo mes-origem (conv. load_shared) ---
rng = np.random.default_rng(0)
t = pd.date_range("2000-01-01", periods=24, freq="MS")   # origens jan/00..dez/01
Y = rng.normal(size=(24, 3, 2)).astype(np.float32)
alvo = xr.DataArray(Y, dims=("time", "lat", "lon"),
                    coords={"time": t, "lat": [0, 1, 2], "lon": [10, 20]})
# fase artificial: nino34 quente se mes-origem em jan..jun, frio jul..dez
n34 = pd.Series(np.where(t.month <= 6, 1.0, -1.0), index=t)

# train_years por ANO-ALVO: [2001] => origens dez/2000..nov/2001
pm = phase_composites(alvo, [2001], n34, warm=0.5, cold=-0.5)
i_feb01 = int(np.where(t == pd.Timestamp("2001-02-01"))[0][0])
i_aug01 = int(np.where(t == pd.Timestamp("2001-08-01"))[0][0])
# alvo mar/2001 (m=3): origem fev/2001 -> quente; alvo set/2001 (m=9): ago -> frio
np.testing.assert_allclose(pm[1][2], Y[i_feb01])
np.testing.assert_allclose(pm[-1][8], Y[i_aug01])
assert np.isnan(pm[0]).all()               # nenhum mes neutro neste cenario
assert np.isnan(pm[1][0]).all()            # jan/2001: origem dez/2000 = frio

# --- enso_offset ---
full = np.zeros((12, 3, 2), np.float32)
off = enso_offset(pm, full, beta=0.5, phase=1)
np.testing.assert_allclose(off[2], 0.5 * Y[i_feb01])
assert off[0].sum() == 0                    # mes sem composito -> offset 0
assert enso_offset(pm, full, 1.0, 0).sum() == 0     # neutro -> zero
assert enso_offset(pm, full, 0.0, 1).sum() == 0     # beta=0 -> baseline

# Series desalinhada e' reindexada ao tempo-origem do alvo
n34_s = pd.Series(1.0, index=pd.date_range("1999-01-01", periods=48, freq="MS"))
pm2 = phase_composites(alvo, [2001], n34_s, 0.5, -0.5)
np.testing.assert_allclose(np.nan_to_num(pm2[1][2]), np.nan_to_num(Y[i_feb01]))

print("test_enso: OK")
