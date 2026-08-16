"""
make_figures.py
---------------
Render the six figures used in the paper from data/hull_resistance.csv and
data/results.json. All figures are saved at 300 dpi into ../figures.
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


def load_csv():
    path = os.path.join(DATA, "hull_resistance.csv")
    names = open(path).readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return names, arr


def load_results():
    with open(os.path.join(DATA, "results.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- fig 1
def fig_correlation(names, arr):
    cols = ["Lwl", "B", "T", "Cb", "V_kn", "L_B", "B_T", "Fn", "R_T_kN"]
    labels = ["$L_{wl}$", "$B$", "$T$", "$C_b$", "$V$",
              "$L/B$", "$B/T$", "$F_n$", "$R_T$"]
    idx = [names.index(c) for c in cols]
    M = arr[:, idx]
    C = np.corrcoef(M, rowvar=False)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(C[i, j]) > 0.55 else "black",
                    fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson $r$")
    fig.savefig(os.path.join(FIG, "fig1_correlation.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_parity(res):
    yt = np.array(res["parity"]["y_true"])
    yp = np.array(res["parity"]["y_pred"])
    m = res["models"][res["best_model"]]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(yt, yp, s=8, alpha=0.35, color=BLUE, edgecolors="none")
    lim = [0, max(yt.max(), yp.max()) * 1.02]
    ax.plot(lim, lim, "--", color=RED, lw=1.2, label="ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True $R_T$ [kN]")
    ax.set_ylabel("Predicted $R_T$ [kN]")
    txt = (f"{res['best_model']}\n$R^2$={m['R2']:.4f}\n"
           f"RMSE={m['RMSE_kN']:.1f} kN\nMAPE={m['MAPE_pct']:.2f}%")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIG, "fig2_parity.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_model_comparison(res):
    order = ["Linear Regression", "k-NN (k=8)", "Random Forest",
             "Gradient Boosting", "MLP (64-64-32)", "Gaussian Process"]
    names = [n for n in order if n in res["models"]]
    r2 = [res["models"][n]["R2"] for n in names]
    mape = [res["models"][n]["MAPE_pct"] for n in names]
    short = ["Linear", "k-NN", "RF", "GBM", "MLP", "GP"][:len(names)]

    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(names))
    b = ax1.bar(x - 0.2, r2, 0.4, color=BLUE, label="$R^2$")
    ax1.set_ylabel("$R^2$", color=BLUE)
    ax1.set_ylim(0.80, 1.005)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_xticks(x); ax1.set_xticklabels(short)
    for xi, v in zip(x - 0.2, r2):
        ax1.text(xi, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, mape, 0.4, color=ORANGE, label="MAPE")
    ax2.set_ylabel("MAPE [%]", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.grid(False)
    for xi, v in zip(x + 0.2, mape):
        ax2.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    fig.savefig(os.path.join(FIG, "fig3_model_comparison.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_importance(res):
    imp = res["importance"]
    lab_map = {"Lwl": "$L_{wl}$", "B": "$B$", "T": "$T$", "Cb": "$C_b$",
               "V_kn": "$V$", "L_B": "$L/B$", "B_T": "$B/T$", "Fn": "$F_n$"}
    items = sorted(imp.items(), key=lambda kv: kv[1])
    labels = [lab_map.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.barh(labels, vals, color=GREEN)
    ax.set_xlabel("Permutation importance\n(decrease in $R^2$, test set)")
    fig.savefig(os.path.join(FIG, "fig4_importance.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig_learning_curve(res):
    lc = res["learning_curve"]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(lc["sizes"], lc["rmse"], "-o", color=BLUE, ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("Training-set size")
    ax.set_ylabel("Hold-out RMSE [kN]")
    ax.set_title("Gradient-boosting learning curve", fontsize=10)
    fig.savefig(os.path.join(FIG, "fig5_learning_curve.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 6
def fig_pareto(res):
    p = res["pareto"]
    cap = np.array(p["cap_all"]) / 1e3
    r = np.array(p["res_all"])
    cf = np.array(p["cap_front"]) / 1e3
    rf_s = np.array(p["res_front_surrogate"])
    rf_o = np.array(p["res_front_oracle"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(cap, r, s=6, alpha=0.18, color="gray", edgecolors="none",
               label="candidate hulls")
    ax.plot(cf, rf_s, "-", color=BLUE, lw=2,
            label="surrogate Pareto front")
    ax.plot(cf, rf_o, "--", color=RED, lw=1.5,
            label="oracle (verification)")
    ax.set_xlabel(r"Transport-capability proxy [$10^3\,\mathrm{m^3\cdot m/s}$]")
    ax.set_ylabel("Total resistance $R_T$ [kN]")
    ax.legend(loc="upper left", fontsize=8)
    txt = (f"screened {p['n_eval']:,} hulls in {p['t_surrogate_s']*1e3:.0f} ms\n"
           f"{p['designs_per_s']:,.0f} designs/s  "
           f"({p['us_per_design']:.1f} $\\mu$s/design)\n"
           f"front error vs oracle: {p['front_mape_pct']:.2f}% MAPE")
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=7.5, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.savefig(os.path.join(FIG, "fig6_pareto.png"))
    plt.close(fig)


def main():
    names, arr = load_csv()
    res = load_results()
    fig_correlation(names, arr)
    fig_parity(res)
    fig_model_comparison(res)
    fig_importance(res)
    fig_learning_curve(res)
    fig_pareto(res)
    print("wrote 6 figures to", FIG)


if __name__ == "__main__":
    main()
