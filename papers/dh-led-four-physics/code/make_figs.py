"""Generate publication figures for the DH-LED paper (#8)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")
from ablation import build, LX, LY  # noqa

SP = "."
FIG = "./figures"
import os; os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130})

res = json.load(open(f"{SP}/ablation.json"))
LABELS = {"homo": "homojunction (Boltzmann, classical)",
          "hetero": "+ heterojunction",
          "+FD": "+ Fermi-Dirac",
          "+QM": "+ Schrodinger-Poisson (full)"}
COL = {"homo": "#888888", "hetero": "#1f77b4", "+FD": "#ff7f0e", "+QM": "#2ca02c"}


def arr(name):
    r = np.array([[x if x is not None else np.nan for x in row]
                  for row in res[name]["rows"]], float)
    return r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4]  # V,I,Irad,IQE,lam


# ---- Fig 1: I-V total + radiative (full model) ----
V, I, Irad, IQE, lam = arr("+QM")
fwd = I > 0
fig, ax = plt.subplots(figsize=(5.2, 4))
ax.semilogy(V[fwd], I[fwd], "-o", ms=4, color="#1f77b4", label="total current $I$")
ax.semilogy(V[fwd], Irad[fwd], "-s", ms=4, color="#2ca02c", label="radiative $I_{rad}=qR_{rad}$")
ax.set_xlabel("forward bias $V$ [V]"); ax.set_ylabel("current [A]")
ax.set_title("GaAs/AlGaAs DH-LED I–V (full four-physics model)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/fig1_iv.png"); plt.close(fig)

# ---- Fig 2: IQE vs current, four-config ablation ----
fig, ax = plt.subplots(figsize=(5.6, 4))
for name in ["homo", "hetero", "+FD", "+QM"]:
    V, I, Irad, IQE, lam = arr(name)
    m = (I > 0) & np.isfinite(IQE)
    ax.semilogx(I[m], IQE[m], "-o", ms=3.5, color=COL[name], label=LABELS[name])
ax.set_xlabel("terminal current $I$ [A]"); ax.set_ylabel("internal quantum efficiency (IQE)")
ax.set_title("IQE vs. current: per-physics ablation")
ax.legend(fontsize=8.5, loc="lower right"); fig.tight_layout()
fig.savefig(f"{FIG}/fig2_iqe_ablation.png"); plt.close(fig)

# ---- Fig 3: I-V ablation (turn-on shift from band offsets) ----
fig, ax = plt.subplots(figsize=(5.6, 4))
for name in ["homo", "hetero", "+FD", "+QM"]:
    V, I, Irad, IQE, lam = arr(name)
    m = I > 0
    ax.semilogy(V[m], I[m], "-o", ms=3.5, color=COL[name], label=LABELS[name])
ax.set_xlabel("forward bias $V$ [V]"); ax.set_ylabel("terminal current $I$ [A]")
ax.set_title("I–V ablation: heterojunction shifts turn-on")
ax.legend(fontsize=8.5, loc="lower right"); fig.tight_layout()
fig.savefig(f"{FIG}/fig3_iv_ablation.png"); plt.close(fig)

# ---- Fig 4: equilibrium carrier confinement + band profile ----
def midline(solver, sol):
    xy = np.asarray(solver.mesh.nodes)      # De Mari-normalised coords
    L = solver.scaling.length               # -> physical metres
    y = xy[:, 1]; uy = np.unique(y); yt = uy[len(uy)//2]
    sel = np.isclose(y, yt, atol=(uy[1]-uy[0])/2)
    xs = xy[sel, 0]; order = np.argsort(xs)
    return (xs[order]*L*1e9, sol.electron_density[sel][order],
            sol.hole_density[sel][order], sol.potential[sel][order])

cfgs = {
    "homo": dict(hetero=False, stats="boltzmann", quantum=None),
    "hetero": dict(hetero=True, stats="boltzmann", quantum=None),
    "+QM": dict(hetero=True, stats="fermi-dirac",
                quantum={"model": "schrodinger", "confinement": "y", "num_states": 4}),
}
prof = {}
for name, kw in cfgs.items():
    s = build(**kw); sol = s.solve_equilibrium()
    prof[name] = midline(s, sol)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
for name, c in [("homo", "#888888"), ("hetero", "#1f77b4"), ("+QM", "#2ca02c")]:
    x, n, p, psi = prof[name]
    axes[0].semilogy(x, n, color=c, label=LABELS[name])
axes[0].axvspan(100, 200, color="gold", alpha=0.15, label="GaAs active")
axes[0].set_xlabel("position x [nm]"); axes[0].set_ylabel("electron density n [m$^{-3}$]")
axes[0].set_title("Equilibrium electron density: active-region confinement")
axes[0].legend(fontsize=8)
for name, c in [("homo", "#888888"), ("hetero", "#1f77b4"), ("+QM", "#2ca02c")]:
    x, n, p, psi = prof[name]
    axes[1].plot(x, psi, color=c, label=LABELS[name])
axes[1].axvspan(100, 200, color="gold", alpha=0.15)
axes[1].set_xlabel("position x [nm]"); axes[1].set_ylabel("electrostatic potential $\\psi$ [V]")
axes[1].set_title("Equilibrium potential profile")
axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_confinement.png"); plt.close(fig)

# ---- Fig 5: electron subband energies (full model) ----
sub = res["+QM"]["subbands"]
fig, ax = plt.subplots(figsize=(4.2, 4))
for i, E in enumerate(sub):
    ax.hlines(E, 0.15, 0.85, color="#2ca02c", lw=2)
    ax.text(0.87, E, f"$E_{i}$ = {E:.3f} eV", va="center", fontsize=9)
ax.set_xlim(0, 1.8); ax.set_xticks([])
ax.set_ylabel("subband energy [eV]")
ax.set_title("Lowest electron subbands\n(10 nm GaAs active layer)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_subbands.png"); plt.close(fig)

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print(" ", f)

# summary table numbers
print("\n=== summary ===")
for name in ["homo", "hetero", "+FD", "+QM"]:
    d = res[name]
    print(f"{name:8s} Vbi={d['Vbi']:.3f}V peak_IQE={d['peak_iqe']:.4f} "
          f"lam={d['wavelength_nm']:.1f}nm")
