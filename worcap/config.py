import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("WORCAP_DATA", ROOT))       # .nc oficiais
OOF_DIR = Path(os.environ.get("WORCAP_OOF", ROOT / "oof"))
SUB_DIR = Path(os.environ.get("WORCAP_SUB", ROOT / "submissions"))
EXT_DIR = Path(os.environ.get("WORCAP_EXT", ROOT / "data_ext"))  # dados externos

TRAIN_START, TRAIN_END = 1940, 2022   # 996 meses
TEST_YEARS = (2023, 2024)             # 24 alvos; 2023 = LB publico, 2024 = privado
HOLDOUT_YEARS = (2018, 2019, 2020, 2021, 2022)  # congelado ate selecao final

N_LAT, N_LON = 301, 261
N_PIXELS = N_LAT * N_LON              # 78.561
N_TEST_ROWS = 24 * N_PIXELS           # 1.885.464

FEATURE_VARS = [
    "t2", "cloud_cover", "surface_pressure", "shum_850", "rel_hum_850",
    "temperature_850", "geopotential_850", "u_850", "v_850",
]
FEATURE_FILES = {
    "t2": "treino_t2.nc",
    "cloud_cover": "treino_cloud_cover.nc",
    "surface_pressure": "treino_surface_pressure.nc",
    "shum_850": "treino_shum_850.nc",
    "rel_hum_850": "treino_rel_hum_850.nc",
    "temperature_850": "treino_temperature_850.nc",
    "geopotential_850": "treino_geopotential_850.nc",
    "u_850": "treino_u_850.nc",
    "v_850": "treino_v_850.nc",
}
TP_FILE = "treino_tp.nc"
TARGET_FILE = "treino_tp_alvo.nc"      # shift M+1 ja aplicado; ultimo mes NaN
TEST_FILE = "teste_features.nc"
SAMPLE_FILE = "sample_submission.csv"
