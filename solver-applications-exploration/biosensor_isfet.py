"""ISFET / BioFET biosensor front-end — analyte detection by threshold shift.

An **ion-sensitive field-effect transistor (ISFET)** and its biological cousin
the **BioFET** are the workhorse transducers of electronic disease detection
(pH, ions, DNA hybridisation, antigen–antibody binding, glucose, …).  Their
physics is a MOS structure whose *metal gate is replaced by an electrolyte and a
functionalised oxide surface*.  When the target analyte binds (or the electrolyte
pH changes), the interface charge / surface dipole shifts the effective gate
potential — which is **exactly the gate work-function term** in DDM.SPC's gate
contact (`u_gate = V/VT + (Phi_ref - Phi_m)/VT`).  So an analyte-induced surface
change of ``dPhi`` [eV] enters the solver as a work-function change of the gate,
and the device response is the resulting shift of the C–V / surface-potential
(and, in a full ISFET, the drain-current) characteristic — the **sensor readout**.

This example builds the ISFET's MOS-capacitor front-end on the built-in
structured mesh (no external mesher needed), sweeps the gate through
accumulation → depletion → inversion for a *baseline* surface and for a surface
carrying *bound analyte* (a small work-function change), and extracts the
threshold shift ``dVth`` — the quantity a real ISFET reports as its signal.

Run:  python examples/biosensor_isfet.py [--plot]
"""
import argparse
import time

import numpy as np

from DDM_SPC import RectangleMeshBuilder, DriftDiffusionSolver, NewtonSolver


# --- geometry: p-Si body (bottom) under a thin gate oxide (top) ---------------
LX, LY = 0.4e-6, 0.5e-6           # 0.4 x 0.5 um cell of the gate stack
NX, NY = 12, 40
Y_OX = 0.45e-6                    # oxide occupies the top 50 nm
NA = 1e23                        # p-body acceptor doping  (1e17 cm^-3)

# Analyte model: each 60 mV of surface-potential change is ~ one pH unit
# (the Nernstian limit).  We represent "no analyte" and "analyte bound" as two
# gate work functions; their difference is the transduced surface signal.
PHI_BASELINE = 4.10              # eV, clean/reference surface
PHI_BOUND = 4.30                 # eV, +0.20 eV after analyte binding
VG_SWEEP = np.round(np.arange(0.0, 2.01, 0.25), 3)


def build_mesh():
    def region_of(xc, yc):
        return "oxide" if yc > Y_OX else "body"

    return RectangleMeshBuilder(LX, LY, NX, NY).build(
        region_of=region_of, region_names=["body", "oxide"])


def surface_potential(mesh, phi_m, vg):
    """Silicon surface potential (normalised u) at gate bias ``vg``.

    A fresh solver per bias point keeps every solve a clean, well-damped cold
    start — robust on the coarse structured mesh and plenty fast for a C–V-style
    sweep.  ``body`` is listed first because the reference region must be a
    semiconductor (the oxide is charge-free).
    """
    solver = DriftDiffusionSolver(
        mesh,
        contacts={
            "gate": {"type": "gate", "region": "oxide",
                     "nodes": mesh.boundary_nodes["top"], "voltage": vg,
                     "work_function": phi_m},
            "body": {"type": "ohmic", "region": "body",
                     "nodes": mesh.boundary_nodes["bottom"], "voltage": 0.0},
        },
        region_properties={
            "body": {"material": "silicon", "doping": -NA},
            "oxide": {"material": "sio2", "doping": 0.0},
        },
        newton=NewtonSolver(step_cap=1.0, max_iterations=200),
        verbose=False,
    )
    sol = solver.solve_bias({"gate": vg, "body": 0.0})

    ys = mesh.nodes[:, 1]
    surf_y = ys[ys <= Y_OX + 1e-12].max()          # top row of the silicon body
    sel = np.where(np.abs(ys - surf_y) < 1e-12)[0]
    return float(np.mean(sol.potential[sel]))       # normalised u, averaged over x


def threshold_voltage(vg, psi_s, u_target):
    """Gate voltage at which the surface potential crosses ``u_target``.

    A simple, monotone proxy for the inversion threshold: linear interpolation
    of the surface-potential-vs-Vg curve at a fixed surface level.
    """
    psi_s = np.asarray(psi_s)
    idx = np.where(psi_s >= u_target)[0]
    if idx.size == 0 or idx[0] == 0:
        return float("nan")
    i = idx[0]
    x0, x1 = vg[i - 1], vg[i]
    y0, y1 = psi_s[i - 1], psi_s[i]
    return float(x0 + (u_target - y0) * (x1 - x0) / (y1 - y0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", action="store_true", help="save a C-V-shift figure")
    args = ap.parse_args()

    mesh = build_mesh()
    t0 = time.time()

    curves = {}
    for label, phi in (("baseline", PHI_BASELINE), ("analyte", PHI_BOUND)):
        curves[label] = np.array([surface_potential(mesh, phi, vg)
                                  for vg in VG_SWEEP])

    # readout threshold: surface potential crossing u = 0 (onset of inversion-side)
    u_target = 0.0
    vth_base = threshold_voltage(VG_SWEEP, curves["baseline"], u_target)
    vth_ana = threshold_voltage(VG_SWEEP, curves["analyte"], u_target)
    dvth = vth_ana - vth_base

    print("ISFET / BioFET biosensor front-end  (DDM.SPC MOS-capacitor solve)")
    print("=" * 66)
    print(f"  p-body doping        : {NA:.0e} m^-3  (1e17 cm^-3)")
    print(f"  analyte surface step : {PHI_BOUND - PHI_BASELINE:+.2f} eV work function")
    print()
    print("  Surface potential  u_s(Vg)   [normalised, u = psi/VT]")
    print("   Vg (V) " + "".join(f"{vg:>9.2f}" for vg in VG_SWEEP))
    for label in ("baseline", "analyte"):
        print(f"   {label:<7}" + "".join(f"{u:>9.2f}" for u in curves[label]))
    print()
    print(f"  Threshold Vg at u_s = {u_target:+.2f}:")
    print(f"    baseline        Vth = {vth_base:.3f} V")
    print(f"    analyte bound   Vth = {vth_ana:.3f} V")
    print(f"    SENSOR RESPONSE dVth = {dvth*1e3:+.0f} mV "
          f"(≈ unity coupling to the {(PHI_BOUND-PHI_BASELINE)*1e3:+.0f} mV surface step)")
    print(f"\n  done in {time.time() - t0:.1f}s")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        ax.plot(VG_SWEEP, curves["baseline"], "o-", label="baseline surface")
        ax.plot(VG_SWEEP, curves["analyte"], "s-",
                label=f"analyte bound (+{(PHI_BOUND-PHI_BASELINE)*1e3:.0f} mV)")
        ax.axhline(u_target, ls=":", c="gray")
        ax.set_xlabel("Gate voltage  $V_G$ (V)")
        ax.set_ylabel("Surface potential  $u_s = \\psi_s/V_T$")
        ax.set_title(f"ISFET transduction: analyte shifts $V_{{th}}$ by "
                     f"{dvth*1e3:+.0f} mV")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        out = "biosensor_isfet.png"
        fig.savefig(out, dpi=130)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
