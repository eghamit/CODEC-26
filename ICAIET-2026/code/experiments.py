"""
experiments.py
--------------
Train and benchmark machine-learning surrogate models that predict the
*residuary resistance per unit weight of displacement* (Rr) of a sailing yacht
from six hull-form parameters, using the **real Delft Systematic Yacht Hull
Series (DSYHS)** towing-tank measurements (UCI "Yacht Hydrodynamics", 308
experiments). See ../data/DATA_SOURCE.md for provenance.

This replaces the earlier synthetic-oracle study: the labels here are physical
measurements, and the models are compared against the traditional polynomial
regression that has historically been fitted to this series.

Because the dataset is small (308 points) and the target grows almost
exponentially with Froude number, we:
  * fit every model on a log1p-transformed target (positive, multiplicative
    error structure), inverting for all reported metrics;
  * evaluate primarily by 10x repeated 10-fold cross-validation (mean +/- sd),
    which is far more reliable than a single split at this sample size;
  * additionally hold out 20% once for the parity plot, permutation importance,
    and a Gaussian-process predictive-interval calibration check.

Outputs: ../data/results.json
"""

import json
import os
import time

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor

SEED = 20270113
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")

FEATURES = ["LCB", "Cp", "L_disp", "B_T", "L_B", "Fn"]
FEAT_LABELS = {
    "LCB": r"LCB", "Cp": r"$C_p$", "L_disp": r"$L/\nabla^{1/3}$",
    "B_T": r"$B/T$", "L_B": r"$L/B$", "Fn": r"$F_n$",
}
TARGET = "Rr"


def load():
    path = os.path.join(DATA_DIR, "yacht_hydrodynamics.data")
    arr = np.loadtxt(path)
    X, y = arr[:, :6], arr[:, 6]
    return X, y


def logtt(estimator):
    """Wrap an estimator so it trains on log1p(target) and inverts for output."""
    return TransformedTargetRegressor(
        regressor=estimator, func=np.log1p, inverse_func=np.expm1)


def build_models():
    return {
        "Linear regression": logtt(
            make_pipeline(StandardScaler(), LinearRegression())),
        "Polynomial reg. (deg 3)": logtt(
            make_pipeline(StandardScaler(),
                          PolynomialFeatures(degree=3, include_bias=False),
                          Ridge(alpha=1.0))),
        "k-NN (k=5)": logtt(
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))),
        "SVR (RBF)": logtt(
            make_pipeline(StandardScaler(),
                          SVR(C=30.0, gamma="scale", epsilon=0.01))),
        "Random forest": logtt(
            RandomForestRegressor(n_estimators=400, min_samples_leaf=1,
                                  n_jobs=-1, random_state=SEED)),
        "Gradient boosting": logtt(
            HistGradientBoostingRegressor(
                max_iter=600, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=SEED)),
        "MLP (64-64)": logtt(
            make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(64, 64),
                                       activation="tanh", alpha=1e-2,
                                       learning_rate_init=3e-3, max_iter=4000,
                                       random_state=SEED))),
        "Gaussian process": logtt(
            make_pipeline(StandardScaler(),
                          GaussianProcessRegressor(
                              kernel=ConstantKernel(1.0)
                              * RBF(length_scale=np.ones(6))
                              + WhiteKernel(noise_level=0.1),
                              normalize_y=True, n_restarts_optimizer=2,
                              random_state=SEED))),
    }


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def medape(a, b):
    return float(np.median(100.0 * np.abs((a - b) / a)))


def evaluate():
    X, y = load()
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=SEED)

    results = {}
    for name, model in build_models().items():
        cv = cross_validate(
            model, X, y, cv=rkf,
            scoring=("r2", "neg_root_mean_squared_error",
                     "neg_mean_absolute_error"),
            n_jobs=-1)
        results[name] = {
            "CV_R2_mean": float(cv["test_r2"].mean()),
            "CV_R2_std": float(cv["test_r2"].std()),
            "CV_RMSE_mean": float(-cv["test_neg_root_mean_squared_error"].mean()),
            "CV_RMSE_std": float(cv["test_neg_root_mean_squared_error"].std()),
            "CV_MAE_mean": float(-cv["test_neg_mean_absolute_error"].mean()),
        }
        print(f"{name:24s} R2={results[name]['CV_R2_mean']:.4f}"
              f"+-{results[name]['CV_R2_std']:.4f} "
              f"RMSE={results[name]['CV_RMSE_mean']:.3f}")

    # ---- single hold-out for parity, importance, timing ------------------
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    best = max(results, key=lambda k: results[k]["CV_R2_mean"])
    best_model = build_models()[best]
    best_model.fit(Xtr, ytr)
    yp = best_model.predict(Xte)
    results[best]["holdout_R2"] = float(r2_score(yte, yp))
    results[best]["holdout_RMSE"] = rmse(yte, yp)

    # inference throughput. At n=308 the GP is both the most accurate model and
    # cheap to query, so it is the natural deployment model; we time it directly.
    big = Xtr[np.random.default_rng(SEED).integers(0, len(Xtr), size=50000)]
    t0 = time.perf_counter()
    for _ in range(5):
        best_model.predict(big)
    infer_us = (time.perf_counter() - t0) / 5 / len(big) * 1e6

    # a fast large-n alternative (gradient boosting) for reference
    gb = build_models()["Gradient boosting"]
    gb.fit(Xtr, ytr)
    t0 = time.perf_counter()
    for _ in range(5):
        gb.predict(big)
    gb_infer_us = (time.perf_counter() - t0) / 5 / len(big) * 1e6

    # permutation importance for best model
    perm = permutation_importance(best_model, Xte, yte, n_repeats=50,
                                  random_state=SEED, n_jobs=-1,
                                  scoring="r2")
    importance = {FEATURES[i]: float(perm.importances_mean[i])
                  for i in range(len(FEATURES))}

    # ---- GP predictive-interval calibration ------------------------------
    calib = gp_calibration(Xtr, ytr, Xte, yte)

    # ---- learning curve (GP, small-data regime) --------------------------
    lc = learning_curve(X, y)

    # ---- design exploration / Pareto (deployed GP surrogate) -------------
    pareto = pareto_exploration(best_model, X, infer_us)

    out = {
        "n_total": int(len(y)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "features": FEATURES, "target": TARGET, "best_model": best,
        "models": results,
        "parity": {"y_true": yte.tolist(), "y_pred": yp.tolist()},
        "importance": importance,
        "calibration": calib,
        "learning_curve": lc,
        "pareto": pareto,
        "infer_us_per_design": float(infer_us),
        "gb_infer_us_per_design": float(gb_infer_us),
    }
    with open(os.path.join(DATA_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nbest (CV R2): {best}")
    print(f"GP 95% interval empirical coverage: {calib['coverage95']*100:.1f}%")
    print(f"GB inference: {infer_us:.2f} us/design")
    print("wrote data/results.json")
    return out


def gp_calibration(Xtr, ytr, Xte, yte):
    """Fit a GP on log target, form 95% predictive intervals, check coverage."""
    scaler = StandardScaler().fit(Xtr)
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=np.ones(6))
        + WhiteKernel(noise_level=0.1),
        normalize_y=True, n_restarts_optimizer=2, random_state=SEED)
    gp.fit(scaler.transform(Xtr), np.log1p(ytr))
    mu, sd = gp.predict(scaler.transform(Xte), return_std=True)
    lo, hi = np.expm1(mu - 1.96 * sd), np.expm1(mu + 1.96 * sd)
    mean_pred = np.expm1(mu)
    cover = np.mean((yte >= lo) & (yte <= hi))
    order = np.argsort(yte)
    return {
        "coverage95": float(cover),
        "y_true": yte[order].tolist(),
        "y_pred": mean_pred[order].tolist(),
        "lo": lo[order].tolist(),
        "hi": hi[order].tolist(),
    }


def learning_curve(X, y):
    sizes = [40, 80, 120, 160, 200, 246]
    rng = np.random.default_rng(SEED)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    out = {"sizes": [], "rmse": []}
    for s in sizes:
        s = min(s, len(Xtr))
        idx = rng.choice(len(Xtr), size=s, replace=False)
        m = build_models()["Gaussian process"]
        m.fit(Xtr[idx], ytr[idx])
        out["sizes"].append(int(s))
        out["rmse"].append(rmse(yte, m.predict(Xte)))
    return out


def pareto_exploration(surrogate, X, infer_us, n=60000, design_fn=0.35):
    """
    Screen hull-form variants at a fixed design Froude number, trading
    displacement capacity against residuary resistance. There is no analytic
    oracle here (labels are physical experiments); fidelity of the front is
    bounded by the surrogate's cross-validated accuracy, and the search is
    confined to the measured parameter envelope (interpolation, not
    extrapolation).
    """
    rng = np.random.default_rng(SEED + 7)
    lo = X[:, :5].min(axis=0)
    hi = X[:, :5].max(axis=0)
    S = lo + rng.uniform(size=(n, 5)) * (hi - lo)
    Fn = np.full((n, 1), design_fn)
    feat = np.hstack([S, Fn])

    rr = surrogate.predict(feat)
    # displacement-length capacity proxy: (L/disp^{1/3})^{-3}  ~  displacement/L^3
    disp_proxy = (S[:, 2]) ** (-3.0)

    front = pareto_front(disp_proxy, rr)  # maximise capacity, minimise Rr
    order = np.argsort(disp_proxy[front])
    front = front[order]
    return {
        "design_fn": design_fn,
        "cap_all": disp_proxy[::40].tolist(),
        "rr_all": rr[::40].tolist(),
        "cap_front": disp_proxy[front].tolist(),
        "rr_front": rr[front].tolist(),
        "n_eval": int(n),
        "infer_us_per_design": float(infer_us),
        "designs_per_s": float(1e6 / infer_us),
    }


def pareto_front(x_max, y_min):
    order = np.argsort(-x_max)
    front, best = [], np.inf
    for i in order:
        if y_min[i] < best:
            best = y_min[i]
            front.append(i)
    return np.array(front)


if __name__ == "__main__":
    evaluate()
