"""V0 — climatologia por (mes, pixel): LOYO+embargo -> OOF -> submissao.

Uso: .venv/bin/python scripts/v0_climatologia.py [--trend]
"""
import sys, time
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from worcap import config, folds, submission
from worcap.oof import save_oof

TREND = "--trend" in sys.argv


def load_tp():
    return xr.open_dataset(config.DATA / config.TP_FILE).tp


def fit_clim_np(data, months, years, train_years, trend=False):
    """clim[m], sigma[m] (e slope[m], y0 se trend) de (time,lat,lon) numpy."""
    mask = np.isin(years, list(train_years))
    clim = np.zeros((12, data.shape[1], data.shape[2]), np.float32)
    sigma = np.zeros_like(clim)
    slope = np.zeros_like(clim)
    yy_all = years[mask].astype(np.float64)
    y0 = yy_all.mean()
    for m in range(1, 13):
        sel = mask & (months == m)
        blk = data[sel]                       # (n_anos, lat, lon)
        yy = years[sel].astype(np.float64)
        if trend:
            X = np.stack([np.ones_like(yy), yy - y0], 1)
            beta = np.linalg.lstsq(X, blk.reshape(len(yy), -1), rcond=None)[0]
            clim[m - 1] = beta[0].reshape(blk.shape[1:])
            slope[m - 1] = beta[1].reshape(blk.shape[1:])
            resid = blk - (X @ beta).reshape(blk.shape)
        else:
            clim[m - 1] = blk.mean(0)
            resid = blk - clim[m - 1]
        sigma[m - 1] = resid.std(0)
    return clim, sigma, slope, y0


def predict_clim(clim, slope, y0, target_years, target_months):
    """clim[m-1] (+ slope*(ano-y0) se trend) para cada alvo."""
    pred = clim[target_months - 1].copy()
    if slope is not None:
        pred += slope[target_months - 1] * (target_years - y0)[:, None, None]
    return pred


def rmse(pred, obs):
    return float(np.sqrt(np.nanmean((pred - obs) ** 2)))


def main():
    t0 = time.time()
    tp = load_tp()
    data = tp.values.astype(np.float32)                 # (996, 301, 261)
    years = tp.time.dt.year.values
    months = tp.time.dt.month.values
    lat, lon = tp.lat.values, tp.lon.values

    # --- sanity: ordem dos ids bate com (time, lat, lon) C-order? ---
    sample_ids = submission.load_sample_index()["id"].values
    exp = [f"{y}_{m:02d}_{la:.2f}_{lo:.2f}"
           for y in config.TEST_YEARS for m in range(1, 13)
           for la in lat for lo in lon]
    assert np.array_equal(sample_ids, np.array(exp)), "ordem de ids NAO bate com (time,lat,lon)"
    print("[ok] ordem de ids = C-order (time, lat, lon)")

    # --- LOYO + embargo ±1 ---
    test = xr.open_dataset(config.DATA / config.TEST_FILE)
    scores, per_month = [], {m: [] for m in range(1, 13)}
    for f in folds.loyo_folds():
        yy = f["test_year"]
        clim, _, slope, y0 = fit_clim_np(data, months, years, f["train_years"], trend=TREND)
        sel = years == yy
        pred = predict_clim(clim, slope if TREND else None, y0,
                            years[sel], months[sel])
        obs = data[sel]
        scores.append(rmse(pred, obs))
        for m in range(1, 13):
            k = np.where(sel & (months == m))[0]
            per_month[m].append(rmse(pred[k[0] - sel.argmax()], obs[k[0] - sel.argmax()]))
        save_oof("v0_clim", yy, xr.DataArray(
            pred, dims=("time", "lat", "lon"),
            coords={"time": tp.time.values[sel], "lat": lat, "lon": lon}))
    print(f"[cv] LOYO RMSE medio = {np.mean(scores):.4f} mm/dia "
          f"(min {np.min(scores):.3f}, max {np.max(scores):.3f})")
    pm = {m: round(np.mean(v), 3) for m, v in per_month.items()}
    print("[cv] RMSE por mes-calendario:", pm)

    # --- persistencia como referencia (prever M+1 = M observado) ---
    pers = [rmse(data[i + 1], data[i]) for i in range(len(data) - 1)]
    print(f"[ref] persistencia RMSE medio = {np.mean(pers):.4f} mm/dia")

    # --- fit final em todos os anos + submissao ---
    clim, sigma, slope, y0 = fit_clim_np(
        data, months, years, folds.final_fit_years(holdout=[]), trend=TREND)
    tt = pd.DatetimeIndex(test.time.values)
    pred = predict_clim(clim, slope if TREND else None, y0,
                        tt.year.values, tt.month.values)
    da = xr.DataArray(pred, dims=("time", "lat", "lon"),
                      coords={"time": test.time.values, "lat": lat, "lon": lon})
    out = config.SUB_DIR / ("v0_clim_trend.csv" if TREND else "v0_clim.csv")
    submission.write_submission(da, out)
    print("[submit]", submission.validate_submission(out), "->", out)
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
