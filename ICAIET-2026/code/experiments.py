"""
experiments.py
--------------
Train and benchmark a family of machine-learning surrogate models that predict
calm-water total resistance R_T from a hull's principal particulars, and
compare them against the semi-empirical oracle in resistance_model.py.

Reported metrics per model:
    R2, RMSE [kN], MAE [kN], MAPE [%]        (hold-out test set)
    R2 (5-fold CV mean +/- std)
    train time [s], inference time per 1k samples [ms]

Also computed:
    * permutation feature importance for the best model,
    * a learning curve (test RMSE vs training-set size),
    * a Pareto exploration (transport efficiency vs resistance) that contrasts
      surrogate-driven search against direct evaluation of the oracle, to
      quantify the speed-up available for early-stage design optimisation.

All results are written to data/results.json for the figure and paper build.
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from resistance_model import BOUNDS, total_resistance

SEED = 20270113
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")

FEATURES = ["Lwl", "B", "T", "Cb", "V_kn", "L_B", "B_T", "Fn"]
TARGET = "R_T_kN"


def load():
    path = os.path.join(DATA_DIR, "hull_resistance.csv")
    names = open(path).readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    col = {n: arr[:, i] for i, n in enumerate(names)}
    X = np.column_stack([col[f] for f in FEATURES])
    y = col[TARGET]
    return X, y


def mape(y_true, y_pred):
    return 100.0 * np.mean(np.abs((y_true - y_pred) / y_true))


def build_models():
    """Return dict name -> (estimator, needs_scaling)."""
    return {
        "Linear Regression": (
            make_pipeline(StandardScaler(), LinearRegression()), True),
        "k-NN (k=8)": (
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=8)),
            True),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=300, max_depth=None, min_samples_leaf=2,
                n_jobs=-1, random_state=SEED), False),
        "Gradient Boosting": (
            HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.05, max_depth=None,
                l2_regularization=1.0, random_state=SEED), False),
        "MLP (64-64-32)": (
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 64, 32), activation="relu",
                    alpha=1e-3, learning_rate_init=2e-3, max_iter=2000,
                    early_stopping=True, n_iter_no_change=25,
                    random_state=SEED)), True),
        "Gaussian Process": (
            make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * RBF(length_scale=np.ones(8))
                    + WhiteKernel(noise_level=1e-2),
                    normalize_y=True, n_restarts_optimizer=0,
                    random_state=SEED)), True),
    }


def evaluate():
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    # Gaussian Process is O(n^3); fit it on a representative 1200-point subset.
    gp_idx = np.random.default_rng(SEED).choice(
        len(Xtr), size=min(1200, len(Xtr)), replace=False)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    fitted = {}

    for name, (model, _scale) in build_models().items():
        if name == "Gaussian Process":
            xtr_f, ytr_f = Xtr[gp_idx], ytr[gp_idx]
        else:
            xtr_f, ytr_f = Xtr, ytr

        t0 = time.perf_counter()
        model.fit(xtr_f, ytr_f)
        t_train = time.perf_counter() - t0

        # inference timing (per 1000 samples), averaged over repeats
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            yp = model.predict(Xte)
        t_inf = (time.perf_counter() - t0) / reps / len(Xte) * 1000 * 1000  # ms/1k

        rmse = np.sqrt(mean_squared_error(yte, yp))
        mae = mean_absolute_error(yte, yp)
        r2 = r2_score(yte, yp)
        mp = mape(yte, yp)

        # cross-validated R2 (skip full CV for GP to keep runtime bounded)
        if name == "Gaussian Process":
            cv_mean, cv_std = float("nan"), float("nan")
        else:
            cv = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
            cv_mean, cv_std = float(cv.mean()), float(cv.std())

        results[name] = {
            "R2": float(r2), "RMSE_kN": float(rmse), "MAE_kN": float(mae),
            "MAPE_pct": float(mp), "CV_R2_mean": cv_mean, "CV_R2_std": cv_std,
            "train_time_s": float(t_train), "infer_ms_per_1k": float(t_inf),
        }
        fitted[name] = model
        print(f"{name:20s} R2={r2:.4f} RMSE={rmse:7.2f} MAPE={mp:5.2f}% "
              f"train={t_train:6.2f}s")

    # ---- best model, parity data, permutation importance -----------------
    best = max(results, key=lambda k: results[k]["R2"])
    best_model = fitted[best]
    yp_best = best_model.predict(Xte)

    perm = permutation_importance(
        best_model, Xte, yte, n_repeats=20, random_state=SEED, n_jobs=-1)
    importance = {FEATURES[i]: float(perm.importances_mean[i])
                  for i in range(len(FEATURES))}

    # ---- learning curve ---------------------------------------------------
    sizes = [100, 250, 500, 1000, 2000, 3000, len(Xtr)]
    lc = {"sizes": [], "rmse": []}
    rng = np.random.default_rng(SEED)
    for s in sizes:
        s = min(s, len(Xtr))
        idx = rng.choice(len(Xtr), size=s, replace=False)
        m = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, l2_regularization=1.0,
            random_state=SEED)
        m.fit(Xtr[idx], ytr[idx])
        lc["sizes"].append(int(s))
        lc["rmse"].append(float(np.sqrt(mean_squared_error(yte, m.predict(Xte)))))

    # ---- Pareto exploration: transport efficiency vs resistance ----------
    # Use a FAST surrogate for batch design screening (gradient boosting),
    # rather than the most accurate but slow Gaussian Process. This reflects
    # the accuracy/latency trade-off a designer actually deploys.
    explore_name = "Gradient Boosting"
    pareto = pareto_exploration(fitted[explore_name], explore_name)

    out = {
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "features": FEATURES, "target": TARGET,
        "models": results, "best_model": best,
        "parity": {"y_true": yte.tolist(), "y_pred": yp_best.tolist()},
        "importance": importance,
        "learning_curve": lc,
        "pareto": pareto,
    }
    with open(os.path.join(DATA_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nmost accurate model: {best}")
    print(f"exploration surrogate: {pareto['surrogate']}")
    print(f"surrogate inference: {pareto['us_per_design']:.2f} us/design "
          f"({pareto['designs_per_s']:,.0f} designs/s)")
    print(f"illustrative acceleration vs {pareto['cfd_hours_assumed']} h "
          f"CFD/design: {pareto['cfd_speedup']:.2e}x")
    print("wrote data/results.json")
    return out


def pareto_exploration(surrogate, surrogate_name, n=60000):
    """
    Sample the design space and build the Pareto front trading total
    resistance against a transport-capability proxy (displacement volume x
    speed).

    The surrogate screens the whole population in a single batched call. We
    report its raw throughput (designs/s) and an *illustrative* acceleration
    factor relative to a representative high-fidelity evaluation cost: a single
    steady RANS resistance computation for a full hull is conservatively of the
    order of a few CPU-hours. This is the regime where surrogates matter --
    exploring 10^4-10^5 candidate hulls is infeasible with direct CFD but takes
    milliseconds with a trained model. The analytic oracle timing is retained
    only as an internal reference; it is a cheap proxy and is not the baseline
    the acceleration factor is quoted against.
    """
    CFD_HOURS = 2.0   # assumed per-design high-fidelity (RANS) cost
    rng = np.random.default_rng(SEED + 7)
    keys = ["Lwl", "B", "T", "Cb", "V_kn"]
    lo = np.array([BOUNDS[k][0] for k in keys])
    hi = np.array([BOUNDS[k][1] for k in keys])
    S = lo + rng.uniform(size=(n, 5)) * (hi - lo)
    Lwl, B, T, Cb, V = S.T
    LB, BT = Lwl / B, B / T
    Fn = (V * 0.514444) / np.sqrt(9.81 * Lwl)
    keep = (LB >= 4.5) & (LB <= 9.5) & (BT >= 2.0) & (BT <= 4.2) & (Fn <= 0.42)
    S, Lwl, B, T, Cb, V = (a[keep] for a in (S, Lwl, B, T, Cb, V))
    Fn = Fn[keep]

    feat = np.column_stack([Lwl, B, T, Cb, V, Lwl / B, B / T, Fn])

    # transport capability proxy: deadweight ~ displaced volume x speed
    volume = Lwl * B * T * Cb          # displaced volume [m^3]
    capability = volume * (V * 0.514444)  # m^3 * m/s (transport rate proxy)

    # surrogate screening (fast)
    t0 = time.perf_counter()
    r_sur = surrogate.predict(feat)
    t_sur = time.perf_counter() - t0

    # oracle evaluation (reference)
    t0 = time.perf_counter()
    r_ora = total_resistance(Lwl, B, T, Cb, V)["R_T_kN"]
    t_ora = time.perf_counter() - t0

    front_idx = pareto_front(capability, r_sur)  # maximise cap, minimise R
    order = np.argsort(capability[front_idx])
    front_idx = front_idx[order]

    n_eval = len(feat)
    us_per_design = t_sur / n_eval * 1e6
    designs_per_s = n_eval / t_sur
    cfd_speedup = (CFD_HOURS * 3600.0) / (t_sur / n_eval)

    # front-fidelity check: error of surrogate front points vs oracle truth
    front_ape = 100.0 * np.abs(
        (r_sur[front_idx] - r_ora[front_idx]) / r_ora[front_idx])

    return {
        "surrogate": surrogate_name,
        "cap_all": capability[::55].tolist(),      # thinned cloud for plotting
        "res_all": r_sur[::55].tolist(),
        "cap_front": capability[front_idx].tolist(),
        "res_front_surrogate": r_sur[front_idx].tolist(),
        "res_front_oracle": r_ora[front_idx].tolist(),
        "front_mape_pct": float(np.mean(front_ape)),
        "n_eval": int(n_eval),
        "t_surrogate_s": float(t_sur),
        "t_oracle_s": float(t_ora),
        "us_per_design": float(us_per_design),
        "designs_per_s": float(designs_per_s),
        "cfd_hours_assumed": CFD_HOURS,
        "cfd_speedup": float(cfd_speedup),
    }


def pareto_front(x_max, y_min):
    """Indices of the non-dominated set: maximise x_max, minimise y_min."""
    order = np.argsort(-x_max)
    front, best_y = [], np.inf
    for i in order:
        if y_min[i] < best_y:
            best_y = y_min[i]
            front.append(i)
    return np.array(front)


if __name__ == "__main__":
    evaluate()
