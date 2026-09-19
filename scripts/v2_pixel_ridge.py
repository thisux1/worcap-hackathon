"""V2: ridge por pixel no residuo do ridge compartilhado (mini-Rodeo).

Uso:
  python scripts/v2_pixel_ridge.py --eval     # subset de anos, ridge vs v2
  python scripts/v2_pixel_ridge.py --submit   # fit 1940-2022 + CSV
"""
import sys
import time
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from worcap.gbm_data import Store
from worcap.pixel_ridge import load_shared, fit_pixel_ridge, eval_year, submit
from worcap.folds import loyo_folds

LAM1, LAM2 = 1000.0, 300.0
EVAL_YEARS = [1983, 1988, 1997, 1998, 2005, 2010, 2015]


def run_eval(years=EVAL_YEARS):
    alvo, Xraw, tgt_month = load_shared()
    Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
    store = Store()
    t0 = time.time()
    folds = {f["test_year"]: f for f in loyo_folds()}
    r1 = {1: [], 2: []}
    r2 = {1: [], 2: []}
    for y in years:
        m = fit_pixel_ridge(store, alvo, Xraw, tgt_month,
                            folds[y]["train_years"], LAM1, LAM2)
        for gap in (1, 2):
            s1, s2 = eval_year(store, alvo, m, y, anchor_gap=gap)
            r1[gap].append(s1)
            r2[gap].append(s2)
            print(f"{y} gap{gap}: ridge {s1:.4f} -> v2 {s2:.4f}", flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
    for gap in (1, 2):
        print(f"media gap{gap}: ridge {np.mean(r1[gap]):.4f} "
              f"-> v2 {np.mean(r2[gap]):.4f}")


if __name__ == "__main__":
    if "--submit" in sys.argv:
        alvo, Xraw, tgt_month = load_shared()
        Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
        store = Store()
        import pandas as pd
        from worcap.config import EXT_DIR
        from worcap.features import production_cols
        df = production_cols(
            pd.read_parquet(EXT_DIR / "features_shared.parquet"))
        p = submit(store, alvo, Xraw, tgt_month, df, LAM1, LAM2,
                   "v2_pixel_ridge")
        print("submission:", p)
    else:
        run_eval()
