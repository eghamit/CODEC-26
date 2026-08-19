"""Validation benchmark for the DDM.SPC device solver against analytic physics.

Establishes the credibility a tools paper needs: the solver reproduces textbook
semiconductor results to quantitative accuracy. Four checks on a 2-D silicon PN
diode (meshing/pn_diode.msh):

  1. Built-in potential   V_bi = V_T ln(N_A N_D / n_i^2)   across doping decades.
  2. Mass action          n p = n_i^2  at thermal equilibrium (max deviation).
  3. Diode ideality       n ~ 1 from the forward-bias exponential (ideal diffusion).
  4. Current conservation I_anode = -I_cathode.

Run:  python benchmark_ddmspc.py [--plot]
"""
import argparse
import os
import time

import numpy as np

from DDM_SPC import DriftDiffusionSolver, MaterialLibrary, read_gmsh

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.join(HERE, "meshing", "pn_diode.msh")


def solver_for(NA_cm3, ND_cm3):
    mesh = read_gmsh(MESH)
    mat = MaterialLibrary().load("silicon")
    s = DriftDiffusionSolver(
        mesh=mesh, material=mat,
        doping={"p-region": -NA_cm3 * 1e6, "n-region": ND_cm3 * 1e6},
        contacts={
            "anode":   {"type": "ohmic", "region": "p-region", "voltage": 0.0},
            "cathode": {"type": "ohmic", "region": "n-region", "voltage": 0.0},
        },
        verbose=False)
    return s, mat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("DDM.SPC validation benchmark vs analytic semiconductor physics")
    print("=" * 66)

    # --- 1. built-in potential across doping decades -----------------------
    print("\n[1] Built-in potential  V_bi = V_T ln(N_A N_D / n_i^2)")
    print(f"  {'N_A=N_D (cm^-3)':>16}{'numeric (V)':>14}{'analytic (V)':>14}{'err (mV)':>10}")
    dopings = [1e15, 1e16, 1e17, 1e18]
    vbi_num, vbi_ana = [], []
    for Ncm in dopings:
        s, mat = solver_for(Ncm, Ncm)
        VT = mat.thermal_voltage
        sol = s.solve_equilibrium()
        num = sol.potential.max() - sol.potential.min()
        ana = VT * np.log((Ncm * 1e6) * (Ncm * 1e6) / mat.ni ** 2)
        vbi_num.append(num); vbi_ana.append(ana)
        print(f"  {Ncm:>16.0e}{num:>14.4f}{ana:>14.4f}{(num-ana)*1e3:>10.1f}")
    vbi_num, vbi_ana = np.array(vbi_num), np.array(vbi_ana)
    max_vbi_err = np.max(np.abs(vbi_num - vbi_ana)) * 1e3
    print(f"  -> max |error| = {max_vbi_err:.1f} mV over {len(dopings)} decades")

    # --- 2. mass action at equilibrium -------------------------------------
    s, mat = solver_for(1e16, 1e16)
    sol = s.solve_equilibrium()
    massact = np.max(np.abs(sol.electron_density * sol.hole_density / mat.ni ** 2 - 1.0))
    print(f"\n[2] Mass action  max|np/n_i^2 - 1| = {massact:.2e}  (machine-precision target)")

    # --- 3. ideality factor + 4. current conservation ----------------------
    VT = mat.thermal_voltage
    V = np.round(np.arange(0.0, 0.66, 0.05), 3)
    # bias the anode, read BOTH terminal currents at each point (same operating
    # state) so conservation is a like-for-like comparison
    s.solve_equilibrium()
    Va, Ia, Ic = [], [], []
    for v in V:
        s.solve_bias({"anode": float(v), "cathode": 0.0})
        Va.append(v)
        Ia.append(s.terminal_current("anode"))
        Ic.append(s.terminal_current("cathode"))
    Va, Ia, Ic = np.array(Va), np.array(Ia), np.array(Ic)
    mask = (Va >= 0.15) & (Va <= 0.45)
    slope = np.polyfit(Va[mask], np.log(np.abs(Ia[mask])), 1)[0]
    ideality = (1.0 / VT) / slope
    fwd = np.abs(Ia) > 1e-20
    cons = np.max(np.abs(Ia[fwd] + Ic[fwd]) / np.abs(Ia[fwd]))
    print(f"\n[3] Ideality factor  n = {ideality:.3f}  (ideal diffusion = 1.000)")
    print(f"[4] Current conservation  max|I_a+I_c|/|I_a| = {cons:.2e}")

    print(f"\nAll checks completed in {time.time()-t0:.0f}s")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
        x = np.log10(dopings)
        a1.plot(x, vbi_ana, "k-", label="analytic $V_T\\ln(N_AN_D/n_i^2)$")
        a1.plot(x, vbi_num, "o", ms=8, mfc="none", label="DDM.SPC")
        a1.set_xlabel("$\\log_{10}(N_A=N_D)$ [cm$^{-3}$]")
        a1.set_ylabel("built-in potential $V_{bi}$ (V)")
        a1.set_title("Built-in potential vs doping"); a1.grid(True); a1.legend()
        a2.semilogy(Va, np.abs(Ia), "o-", label="DDM.SPC")
        ideal = np.abs(Ia[mask][0]) * np.exp((Va - Va[mask][0]) / VT)
        a2.semilogy(Va, ideal, "k--", label="ideal $\\exp(V/V_T)$")
        a2.set_ylim(np.abs(Ia[Ia!=0]).min()*0.5, np.abs(Ia).max()*2)
        a2.set_xlabel("forward bias $V$ (V)"); a2.set_ylabel("|I| (A)")
        a2.set_title(f"Forward I-V, ideality $n$ = {ideality:.3f}")
        a2.grid(True, which="both"); a2.legend()
        fig.tight_layout(); fig.savefig("benchmark_ddmspc.png", dpi=130)
        print("  saved benchmark_ddmspc.png")


if __name__ == "__main__":
    main()
