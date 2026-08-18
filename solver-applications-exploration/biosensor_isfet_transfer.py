"""Full ISFET / BioFET biosensor — drain-current transfer curves I_D(V_G).

This is the complete-device companion to ``biosensor_isfet.py`` (which solved
only the MOS-capacitor front-end).  Here the whole ISFET is solved on the tuned
short-channel MOSFET mesh (``meshing/mosfet_v3.msh``): n+ source / drain, p
body, insulated gate over the oxide.  The **analyte** is modelled the way an
ISFET actually transduces it -- a change in the functionalised-oxide surface
charge / dipole shifts the effective gate potential, which is the gate
**work-function** term in DDM.SPC (``u_gate = V/VT + (Phi_ref - Phi_m)/VT``).

For a series of analyte-induced surface steps ``dPhi`` we compute the transfer
characteristic I_D(V_G) at a fixed small drain bias (linear region), then extract
the two figures of merit a real ISFET assay is characterised on:

* **threshold-voltage shift**  ``dVth``  -- the sensor signal (a calibration
  line ``dVth`` vs. ``dPhi`` is the device's sensitivity), and
* **sub-threshold swing**  ``SS`` [mV/decade] -- how steeply the device turns on,
  which sets the smallest resolvable surface change (the noise-free detection
  limit).

Run:  python examples/biosensor_isfet_transfer.py [--plot] [--depth 1e-6]

Note: this is a *device-physics* transfer-curve model.  The electrolyte double
layer (Debye screening) and the site-binding surface chemistry that make the
coupling sub-unity are not modelled here -- ``dPhi`` is taken as the net surface
potential that reaches the gate.  See docs/applications_exploration.md (extension
G-DL) for that next layer.
"""
import argparse
import time

import numpy as np

from DDM_SPC import read_gmsh, DriftDiffusionSolver

MESH = "meshing/mosfet_v3.msh"
ND, NA = 1e26, 1e23                       # n+ S/D (1e20), p-body (1e17) [m^-3]
PHI_REF_GATE = 4.10                       # clean-surface gate work function [eV]
VD_LIN = 0.10                             # linear-region drain bias [V]
VG_MIN, VG_MAX, VG_STEP = -0.6, 2.0, 0.1  # gate sweep [V]


def build_solver(mesh, phi_gate, device_depth):
    return DriftDiffusionSolver(
        mesh,
        contacts={
            "drain":  {"type": "ohmic", "region": "n+drain", "voltage": 0.0},
            "gate":   {"type": "gate", "region": "oxide", "voltage": 0.0,
                       "work_function": phi_gate},
            "source": {"type": "ohmic", "region": "n+source", "voltage": 0.0},
            "body":   {"type": "ohmic", "region": "p+substrate", "voltage": 0.0},
        },
        region_properties={
            "n+drain":     {"material": "silicon", "doping": ND},
            "oxide":       {"material": "sio2", "doping": 0.0},
            "n+source":    {"material": "silicon", "doping": ND},
            "p+substrate": {"material": "silicon", "doping": -NA},
        },
        recombination="srh+auger",
        device_depth=device_depth,
        verbose=False,
    )


def transfer_curve(mesh, phi_gate, device_depth):
    """I_D(V_G) at fixed V_D, warm-started for robustness."""
    solver = build_solver(mesh, phi_gate, device_depth)
    solver.solve_equilibrium()
    # ramp the drain up at V_G = 0, then walk V_G down to the sweep floor
    for vd in np.arange(0.02, VD_LIN + 1e-9, 0.02):
        solver.solve_bias({"gate": 0.0, "drain": float(vd)})
    for vg in np.arange(-0.1, VG_MIN - 1e-9, -0.1):
        solver.solve_bias({"gate": float(vg), "drain": VD_LIN})
    # sweep V_G up, recording the drain current
    vgs = np.round(np.arange(VG_MIN, VG_MAX + 1e-9, VG_STEP), 3)
    ids = []
    for vg in vgs:
        solver.solve_bias({"gate": float(vg), "drain": VD_LIN})
        ids.append(abs(solver.terminal_current("drain")))
    return vgs, np.array(ids)


def extract_vth_constant_current(vgs, ids, i_th):
    """Threshold = V_G where I_D crosses a fixed criterion current (log-interp)."""
    for i in range(1, len(ids)):
        if ids[i - 1] < i_th <= ids[i]:
            l0, l1 = np.log10(ids[i - 1]), np.log10(ids[i])
            return float(vgs[i - 1] + (np.log10(i_th) - l0)
                         * (vgs[i] - vgs[i - 1]) / (l1 - l0))
    return float("nan")


def subthreshold_swing(vgs, ids, i_lo, i_hi):
    """SS [mV/dec] from a least-squares fit of log10(I_D) vs V_G in weak inversion."""
    m = (ids >= i_lo) & (ids <= i_hi)
    if m.sum() < 2:
        return float("nan")
    slope = np.polyfit(vgs[m], np.log10(ids[m]), 1)[0]   # decades per volt
    return float(1e3 / slope) if slope > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", action="store_true", help="save transfer-curve figure")
    ap.add_argument("--depth", type=float, default=1e-6,
                    help="out-of-plane device width W [m] (default 1 um)")
    ap.add_argument("--analyte-steps", type=float, nargs="+",
                    default=[0.0, 0.10, 0.20],
                    help="analyte-induced surface steps dPhi [eV]")
    args = ap.parse_args()

    mesh = read_gmsh(MESH)
    t0 = time.time()

    curves = {}
    for dphi in args.analyte_steps:
        vgs, ids = transfer_curve(mesh, PHI_REF_GATE + dphi, args.depth)
        curves[dphi] = (vgs, ids)
        print(f"  computed transfer curve  dPhi = {dphi:+.2f} eV "
              f"({time.time() - t0:.0f}s)")

    # figures of merit, referenced to the clean (dPhi = 0) surface
    ref_dphi = args.analyte_steps[0]
    vgs0, ids0 = curves[ref_dphi]
    i_th = 10 ** ((np.log10(ids0.min()) + np.log10(ids0.max())) / 2)  # geo-mean criterion
    ss_lo, ss_hi = ids0.min() * 3, i_th                              # weak-inversion band

    print("\nFull ISFET / BioFET transfer-curve analysis  (DDM.SPC, mosfet_v3 mesh)")
    print("=" * 72)
    print(f"  W = {args.depth:.1e} m,  V_D = {VD_LIN:.2f} V,  I_th = {i_th:.2e} A")
    print(f"  {'dPhi (eV)':>10}{'Vth (V)':>10}{'dVth (mV)':>12}"
          f"{'SS (mV/dec)':>13}{'Ion (A)':>12}")
    vth_ref = None
    rows = []
    for dphi in args.analyte_steps:
        vgs, ids = curves[dphi]
        vth = extract_vth_constant_current(vgs, ids, i_th)
        ss = subthreshold_swing(vgs, ids, ss_lo, ss_hi)
        ion = ids[-1]
        if vth_ref is None:
            vth_ref = vth
        dvth = (vth - vth_ref) * 1e3
        rows.append((dphi, vth, dvth, ss, ion))
        print(f"  {dphi:>10.2f}{vth:>10.3f}{dvth:>12.0f}{ss:>13.1f}{ion:>12.2e}")

    # sensitivity = slope of dVth vs dPhi
    dphis = np.array([r[0] for r in rows])
    dvths = np.array([r[2] for r in rows]) / 1e3
    if len(dphis) >= 2:
        sens = np.polyfit(dphis, dvths, 1)[0]
        print(f"\n  SENSITIVITY  dVth/dPhi = {sens:.3f} V/V "
              f"(ideal = 1.00; a real ISFET is sub-unity from double-layer coupling)")
    print(f"  done in {time.time() - t0:.0f}s")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.4))
        for dphi in args.analyte_steps:
            vgs, ids = curves[dphi]
            lbl = f"$\\Delta\\Phi$ = {dphi:+.2f} eV"
            axl.plot(vgs, ids * 1e6, "-o", ms=3, label=lbl)
            axr.semilogy(vgs, ids, "-o", ms=3, label=lbl)
        axl.set_xlabel("Gate voltage $V_G$ (V)")
        axl.set_ylabel("Drain current $I_D$ ($\\mu$A)")
        axl.set_title("Transfer curve (linear)")
        axl.grid(True); axl.legend()
        axr.axhline(i_th, ls=":", c="gray", label="$I_{th}$")
        axr.set_xlabel("Gate voltage $V_G$ (V)")
        axr.set_ylabel("Drain current $I_D$ (A)")
        axr.set_title("Transfer curve (semilog) — analyte shifts $V_{th}$")
        axr.grid(True, which="both"); axr.legend()
        fig.tight_layout()
        out = "biosensor_isfet_transfer.png"
        fig.savefig(out, dpi=130)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
