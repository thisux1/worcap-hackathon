"""V0.5: ridge por pixel sobre features mensais compartilhadas.

Alvo: z = (tp_alvo - clim_fold[mes_alvo]) / sigma_fold[mes_alvo].
X (meses x K) e' igual para todos os pixels -> W = (X'X + lam*I)^-1 X'Z
resolve os 78.561 pixels de uma vez (MultiLLR-lite).

Uso: python scripts/v05_ridge.py
"""
import sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from worcap.config import DATA, TP_FILE, TARGET_FILE, EXT_DIR
from worcap.folds import loyo_folds, final_fit_years
from worcap.climatology import fit_climatology
from worcap.oof import save_oof
from worcap.submission import write_submission

LAMBDAS = [300.0, 1000.0, 3000.0]
LAMBDA_OOF = 1000.0    # lambda gravado no OOF store
VIA = "v05_ridge"


def load():
    tp = xr.open_dataset(DATA / TP_FILE)["tp"]
    alvo = xr.open_dataset(DATA / TARGET_FILE)["tp_alvo"]
    from worcap.features import production_cols
    df = production_cols(pd.read_parquet(EXT_DIR / "features_shared.parquet"))
    X = df.reindex(pd.DatetimeIndex(alvo.time.values)).fillna(0.0)
    tgt_month = pd.DatetimeIndex(alvo.time.values) + pd.offsets.MonthBegin(1)
    return tp, alvo, X.values.astype(np.float64), tgt_month, df


def ridge_fit(X, Z, lam):
    p = np.eye(X.shape[1]) * lam
    p[0, 0] = 0.0                      # bias nao penalizado
    return np.linalg.solve(X.T @ X + p, X.T @ Z)


def predict_rows(X, W, rows, c, s, shape):
    zh = (X[rows] @ W).reshape(rows.sum(), *shape[1:])
    return np.clip(c[rows] + s[rows] * zh, 0, None)


def run_cv(lams=LAMBDAS):
    tp, alvo, Xraw, tgt_month, _ = load()
    Y = alvo.values
    tgt_year = tgt_month.year.values
    tgt_mnum = tgt_month.month.values
    Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
    valid = ~np.isnan(Y[:, 0, 0])
    scores = {l: [] for l in lams}

    for f in loyo_folds():
        ty = f["test_year"]
        rows_te = valid & (tgt_year == ty)
        rows_tr = valid & np.isin(tgt_year, f["train_years"])
        clim, sigma = fit_climatology(tp, f["train_years"])
        c = clim.values[tgt_mnum - 1]
        s = np.maximum(sigma.values[tgt_mnum - 1], 1e-3)
        Ztr = (Y[rows_tr] - c[rows_tr]) / s[rows_tr]
        Ztr = np.nan_to_num(Ztr.reshape(rows_tr.sum(), -1))

        mu = Xraw[rows_tr].mean(0)
        sd = Xraw[rows_tr].std(0); sd[sd == 0] = 1
        mu[0], sd[0] = 0.0, 1.0
        Xn = (Xraw - mu) / sd
        Xtr, Xte = Xn[rows_tr], Xn[rows_te]

        line = f"fold {ty}:"
        for l in lams:
            W = ridge_fit(Xtr, Ztr, l)
            yh = predict_rows(Xn, W, rows_te, c, s, Y.shape)
            rmse = np.sqrt(np.nanmean((yh - Y[rows_te]) ** 2))
            scores[l].append(rmse)
            line += f" {l:g}={rmse:.4f}"
            if l == LAMBDA_OOF:
                da = xr.DataArray(
                    yh, dims=("time", "lat", "lon"),
                    coords={"time": tgt_month.values[rows_te],
                            "lat": alvo.lat, "lon": alvo.lon})
                save_oof(VIA, ty, da)
        print(line, flush=True)

    print("\nOOF RMSE medio por lambda:")
    for l in lams:
        print(f"  {l:>6g}: {np.mean(scores[l]):.4f}")


def submit(lam=LAMBDA_OOF, tag="v05_ridge", all_years=False):
    """Fit em final_fit_years (ou todos, all_years=True) + CSV validado."""
    from worcap.config import TEST_FILE, SUB_DIR, TRAIN_START, TRAIN_END
    tp, alvo, Xraw, tgt_month, df = load()
    Y = alvo.values
    tgt_year = tgt_month.year.values
    tgt_mnum = tgt_month.month.values
    Xraw = np.hstack([np.ones((len(Xraw), 1)), Xraw])
    valid = ~np.isnan(Y[:, 0, 0])
    years = (list(range(TRAIN_START, TRAIN_END + 1)) if all_years
             else final_fit_years())
    rows_tr = valid & np.isin(tgt_year, years)
    clim, sigma = fit_climatology(tp, years)
    c = clim.values[tgt_mnum - 1]
    s = np.maximum(sigma.values[tgt_mnum - 1], 1e-3)
    Ztr = np.nan_to_num(((Y[rows_tr] - c[rows_tr]) / s[rows_tr])
                        .reshape(rows_tr.sum(), -1))
    mu = Xraw[rows_tr].mean(0)
    sd = Xraw[rows_tr].std(0); sd[sd == 0] = 1
    mu[0], sd[0] = 0.0, 1.0
    W = ridge_fit((Xraw[rows_tr] - mu) / sd, Ztr, lam)

    test = xr.open_dataset(DATA / TEST_FILE)
    orig = pd.DatetimeIndex(test.time_origem.values)
    T = pd.DatetimeIndex(test.time.values)
    Xte = df.reindex(orig).fillna(0.0).values.astype(np.float64)
    Xte = np.hstack([np.ones((len(Xte), 1)), Xte])
    zh = ((Xte - mu) / sd) @ W                       # (24, npix)
    yh = np.stack([np.clip(clim.values[t.month - 1]
                           + np.maximum(sigma.values[t.month - 1], 1e-3)
                           * zh[j].reshape(301, 261), 0, None)
                   for j, t in enumerate(T)])
    da = xr.DataArray(yh, dims=("time", "lat", "lon"),
                      coords={"time": T, "lat": alvo.lat, "lon": alvo.lon})
    path = SUB_DIR / f"{tag}.csv"
    write_submission(da, path)
    print("submission:", path)


if __name__ == "__main__":
    if "--submit" in sys.argv:
        all_y = "--all" in sys.argv
        submit(tag="v05_ridge_all" if all_y else "v05_ridge", all_years=all_y)
    else:
        run_cv()
