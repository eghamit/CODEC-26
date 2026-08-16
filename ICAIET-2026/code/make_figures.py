"""
make_figures.py
---------------
Render the figures used in the paper from the real DSYHS dataset
(../data/yacht_hydrodynamics.data) and the experiment results
(../data/results.json). Figures are saved at 300 dpi into ../figures.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.3, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
BLUE, ORANGE, GREEN, RED = "#1f5c99", "#e07b1a", "#2a8c3c", "#c0392b"

SHORT = {
    "Linear regression": "Linear", "Polynomial reg. (deg 3)": "Poly-3",
    "k-NN (k=5)": "k-NN", "SVR (RBF)": "SVR", "Random forest": "RF",
    "Gradient boosting": "GBM", "MLP (64-64)": "MLP",
    "Gaussian process": "GP",
}
ORDER = ["Linear regression", "Polynomial reg. (deg 3)", "k-NN (k=5)",
         "SVR (RBF)", "Random forest", "Gradient boosting", "MLP (64-64)",
         "Gaussian process"]
LAB = {"LCB": "LCB", "Cp": r"$C_p$", "L_disp": r"$L/\nabla^{1/3}$",
       "B_T": r"$B/T$", "L_B": r"$L/B$", "Fn": r"$F_n$"}


def load_results():
    with open(os.path.join(DATA, "results.json")) as f:
        return json.load(f)


# ------------------------------------------------------------------ fig 1
def fig_data():
    arr = np.loadtxt(os.path.join(DATA, "yacht_hydrodynamics.data"))
    Fn, Ld, Rr = arr[:, 5], arr[:, 2], arr[:, 6]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    sc = ax.scatter(Fn, Rr, c=Ld, cmap="viridis", s=22, edgecolors="k",
                    linewidths=0.3)
    ax.set_xlabel(r"Froude number $F_n$")
    ax.set_ylabel(r"Residuary resistance $R_r$ (per unit weight)")
    ax.set_title("Delft Systematic Yacht Hull Series (308 experiments)",
                 fontsize=10)
    fig.colorbar(sc, ax=ax, label=r"$L/\nabla^{1/3}$")
    fig.savefig(os.path.join(FIG, "fig1_data.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig 2
def fig_parity(res):
    yt = np.array(res["parity"]["y_true"])
    yp = np.array(res["parity"]["y_pred"])
    m = res["models"][res["best_model"]]
    fig, ax = plt.subplots(figsize=(4.5, 4.3))
    ax.scatter(yt, yp, s=26, alpha=0.6, color=BLUE, edgecolors="k",
               linewidths=0.3)
    lim = [min(yt.min(), yp.min()) - 1, max(yt.max(), yp.max()) * 1.03]
    ax.plot(lim, lim, "--", color=RED, lw=1.2, label="ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"Measured $R_r$")
    ax.set_ylabel(r"Predicted $R_r$")
    txt = (f"{res['best_model']}\nhold-out $R^2$="
           f"{m.get('holdout_R2', float('nan')):.4f}\n"
           f"RMSE={m.get('holdout_RMSE', float('nan')):.3f}")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIG, "fig2_parity.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig_model_comparison(res):
    names = [n for n in ORDER if n in res["models"]]
    r2 = [res["models"][n]["CV_R2_mean"] for n in names]
    r2e = [res["models"][n]["CV_R2_std"] for n in names]
    rm = [res["models"][n]["CV_RMSE_mean"] for n in names]
    rme = [res["models"][n]["CV_RMSE_std"] for n in names]
    short = [SHORT[n] for n in names]
    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(6.6, 3.9))
    ax1.bar(x - 0.2, r2, 0.4, yerr=r2e, capsize=2, color=BLUE, label="$R^2$")
    ax1.set_ylabel("CV $R^2$", color=BLUE)
    ax1.set_ylim(min(r2) - 0.06, 1.005)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_xticks(x); ax1.set_xticklabels(short, rotation=0)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, rm, 0.4, yerr=rme, capsize=2, color=ORANGE, label="RMSE")
    ax2.set_ylabel(r"CV RMSE ($R_r$ units)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.grid(False)
    fig.savefig(os.path.join(FIG, "fig3_model_comparison.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig 4
def fig_importance(res):
    imp = res["importance"]
    items = sorted(imp.items(), key=lambda kv: kv[1])
    labels = [LAB.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.barh(labels, vals, color=GREEN)
    ax.set_xlabel("Permutation importance\n(decrease in $R^2$, hold-out set)")
    fig.savefig(os.path.join(FIG, "fig4_importance.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig 5
def fig_learning_curve(res):
    lc = res["learning_curve"]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(lc["sizes"], lc["rmse"], "-o", color=BLUE, ms=5)
    ax.set_xlabel("Training-set size")
    ax.set_ylabel(r"Hold-out RMSE ($R_r$ units)")
    ax.set_title("Gaussian-process learning curve", fontsize=10)
    fig.savefig(os.path.join(FIG, "fig5_learning_curve.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig 6
def fig_calibration(res):
    c = res["calibration"]
    yt = np.array(c["y_true"]); yp = np.array(c["y_pred"])
    lo = np.array(c["lo"]); hi = np.array(c["hi"])
    idx = np.arange(len(yt))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.fill_between(idx, lo, hi, color=BLUE, alpha=0.2,
                    label="95% predictive interval")
    ax.plot(idx, yp, "-", color=BLUE, lw=1.2, label="GP mean")
    ax.scatter(idx, yt, s=16, color=RED, zorder=5, label="measured")
    ax.set_xlabel("Hold-out sample (sorted by measured $R_r$)")
    ax.set_ylabel(r"Residuary resistance $R_r$")
    ax.text(0.03, 0.95, f"empirical coverage = {c['coverage95']*100:.1f}%\n"
                        f"(nominal 95%)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(FIG, "fig6_calibration.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig 7
def fig_pareto(res):
    p = res["pareto"]
    cap = np.array(p["cap_all"]); rr = np.array(p["rr_all"])
    cf = np.array(p["cap_front"]); rf = np.array(p["rr_front"])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(cap, rr, s=6, alpha=0.15, color="gray", edgecolors="none",
               label="candidate hulls")
    ax.plot(cf, rf, "-o", color=BLUE, lw=2, ms=3, label="surrogate Pareto front")
    ax.set_xlabel(r"Displacement-capacity proxy $(L/\nabla^{1/3})^{-3}$")
    ax.set_ylabel(r"Predicted residuary resistance $R_r$")
    ax.set_title(rf"Design exploration at $F_n={p['design_fn']}$", fontsize=10)
    txt = (f"screened {p['n_eval']:,} hulls\n"
           f"{p['designs_per_s']:,.0f} designs/s "
           f"({p['infer_us_per_design']:.1f} $\\mu$s each)")
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=7.5, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(FIG, "fig7_pareto.png"))
    plt.close(fig)


def main():
    res = load_results()
    fig_data()
    fig_parity(res)
    fig_model_comparison(res)
    fig_importance(res)
    fig_learning_curve(res)
    fig_calibration(res)
    fig_pareto(res)
    print("wrote 7 figures to", FIG)


if __name__ == "__main__":
    main()
