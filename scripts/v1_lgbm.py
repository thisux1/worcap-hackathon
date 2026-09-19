"""V1 v0: LightGBM global (multi-tarefa) sobre anomalia padronizada z.

- Treino: meses dos anos de treino do fold, ~n_pix pixels/mes, lag augmentation.
- Avaliacao em 2 regimes reais: ancora dez/(Y-1) (lags 1-12, tipo 2023) e
  ancora dez/(Y-2) (lags 13-24, tipo 2024).
- Reconstrucao: y = clim + sigma * z_hat, clip >= 0.

Uso:
  python scripts/v1_lgbm.py --years 1997 2010 2015   # avalia anos especificos
"""
import sys, time
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from worcap.config import DATA, TP_FILE, TARGET_FILE, EXT_DIR, SUB_DIR
from worcap.folds import loyo_folds, final_fit_years
from worcap.climatology import fit_climatology
from worcap.gbm_data import (Store, build_train_matrix, build_eval_matrix,
                             build_test_matrix)
from worcap.submission import write_submission

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
              min_data_in_leaf=200, num_threads=12, verbose=-1)
N_ROUNDS = 400
N_PIX = 5000


def predict_year(mdl, Xe, metas, npix, shape):
    zh = mdl.predict(Xe)
    yh, k = [], 0
    for m in metas:
        blk = zh[k:k + npix].reshape(shape)
        yh.append(np.clip(m["c"] + m["s"] * blk, 0, None))
        k += npix
    return np.stack(yh)


def fit_and_eval(years_eval, n_pix=N_PIX, rounds=N_ROUNDS, seed=0):
    t0 = time.time()
    print("abrindo store (lazy)...", flush=True)
    store = Store()
    alvo = xr.open_dataset(DATA / TARGET_FILE)["tp_alvo"]
    shared = pd.read_parquet(EXT_DIR / "features_shared.parquet")
    folds = {f["test_year"]: f for f in loyo_folds()}
    print(f"store pronto ({time.time()-t0:.0f}s)", flush=True)

    res = []
    for y in years_eval:
        f = folds[y]
        t0 = time.time()
        clim, sigma = fit_climatology(store.das["tp"], f["train_years"])
        X, z, yrs = build_train_matrix(store, alvo, clim, sigma, shared,
                                       f["train_years"], n_pix=n_pix,
                                       seed=seed)
        print(f"  {y}: matriz {X.shape} ({time.time()-t0:.0f}s)", flush=True)
        vy = max(f["train_years"])           # ano interno p/ early stop
        n_val = int((yrs == vy).sum())       # rows do ultimo ano vao por ultimo
        dtr = lgb.Dataset(X.iloc[:-n_val], z[:-n_val])
        dva = lgb.Dataset(X.iloc[-n_val:], z[-n_val:], reference=dtr)
        mdl = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(80, verbose=False)])
        print(f"  {y}: best_iter={mdl.best_iteration}", flush=True)
        del X, z
        npix, shape = store.npix, (store.das["tp"].sizes["lat"],
                                   store.das["tp"].sizes["lon"])
        for gap, tag in [(1, "lag1-12"), (2, "lag13-24")]:
            try:
                Xe, metas, yt = build_eval_matrix(store, alvo, clim, sigma,
                                                  shared, y, anchor_gap=gap)
            except IndexError:
                continue
            yh = predict_year(mdl, Xe, metas, npix, shape)
            rmse = np.sqrt(np.nanmean((yh - yt) ** 2))
            res.append({"year": y, "regime": tag, "rmse": rmse})
            print(f"  {y} {tag}: RMSE={rmse:.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    df = pd.DataFrame(res)
    print("\n", df.pivot(index="year", columns="regime", values="rmse"))
    print("media:", df.groupby("regime").rmse.mean().to_dict())
    return df


def submit(n_pix=N_PIX, rounds=N_ROUNDS, seed=0, tag="v1_lgbm"):
    """Fit em final_fit_years (sem holdout) + previsao dos 24 alvos + CSV."""
    store = Store()
    alvo = xr.open_dataset(DATA / TARGET_FILE)["tp_alvo"]
    shared = pd.read_parquet(EXT_DIR / "features_shared.parquet")
    years = final_fit_years()
    clim, sigma = fit_climatology(store.das["tp"], years)
    X, z, yrs = build_train_matrix(store, alvo, clim, sigma, shared, years,
                                   n_pix=n_pix, seed=seed)
    print(f"matriz final {X.shape}", flush=True)
    vy = max(years)
    n_val = int((yrs == vy).sum())
    dtr = lgb.Dataset(X.iloc[:-n_val], z[:-n_val])
    dva = lgb.Dataset(X.iloc[-n_val:], z[-n_val:], reference=dtr)
    mdl = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(80, verbose=False)])
    print(f"best_iter={mdl.best_iteration}", flush=True)
    del X, z
    Xe, metas = build_test_matrix(store, clim, sigma, shared)
    yh = predict_year(mdl, Xe, metas, store.npix,
                      (store.das["tp"].sizes["lat"], store.das["tp"].sizes["lon"]))
    da = xr.DataArray(
        yh, dims=("time", "lat", "lon"),
        coords={"time": [m["time"] for m in metas],
                "lat": store.das["tp"].lat, "lon": store.das["tp"].lon})
    path = SUB_DIR / f"{tag}.csv"
    out = write_submission(da, path)
    print("submission:", out)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--submit" in args:
        submit()
    else:
        yi = args.index("--years")
        years = [int(a) for a in args[yi + 1:]]
        fit_and_eval(years)
