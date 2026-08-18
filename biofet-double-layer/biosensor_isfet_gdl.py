"""ISFET / BioFET with the electrolyte double-layer (G-DL) front-end.

This closes the gap flagged in the earlier ideal-coupling examples: the bare
device solver gives ``dVth/dPhi = 1.000`` (unity coupling) because it has no
electrolyte.  Here the ``double_layer`` module supplies the missing physics --
site-binding surface chemistry + a Gouy-Chapman-Stern double layer -- so the
analyte-to-gate coupling becomes realistically **sub-unity**:

  analyte (pH / bound charge, ionic strength)
        |  double_layer:  site-binding + Gouy-Chapman-Stern
        v
  surface potential  psi_0   (sub-Nernstian dpsi_0/dpH < 59 mV/pH)
        |  enters the gate as an effective work-function shift  dPhi = -psi_0
        v
  DDM.SPC device solve  ->  I_D(V_G) transfer curve  ->  Vth  ->  sensor signal

Two studies are produced:

1. **pH response (ISFET).** For each oxide, the electrolyte layer's sub-Nernstian
   dpsi_0/dpH is propagated through the device to a threshold-voltage sensitivity
   dVth/dpH -- below the 59.5 mV/pH Nernst limit, unlike the ideal-coupling model.

2. **BioFET Debye screening.** A fixed bound-biomolecule charge is detected at a
   set distance; sweeping the ionic strength shows the response collapse once the
   Debye length falls below that distance -- the fundamental BioFET detection limit.

By default only the (fast) electrolyte-layer analysis runs.  ``--device`` also
runs the DDM.SPC transfer-curve solve at a few pH points to demonstrate the full
chain end-to-end (slower; needs gmsh + the mosfet_v3 mesh).

Run:  python biosensor_isfet_gdl.py [--plot] [--device]
"""
import argparse
import time

import numpy as np

import double_layer as dl

NERNST = 2.303 * dl.thermal_voltage() * 1e3          # 59.2 mV/pH at 298 K


# ---------------------------------------------------------------- electrolyte
def ph_response(oxides, pH_grid, ionic_strength):
    """psi_0(pH) curves and the local sensitivity for each oxide."""
    out = {}
    for name in oxides:
        ox = dl.OXIDES[name]
        psi = np.array([ox.surface_potential(pH, ionic_strength) for pH in pH_grid])
        sens = np.gradient(psi, pH_grid) * 1e3          # mV/pH
        out[name] = (psi, sens)
    return out


def debye_screening(sigma_bio, distance_nm, I_grid):
    """Bound-charge surface-potential response vs ionic strength (BioFET limit)."""
    d = distance_nm * 1e-9
    lam = np.array([dl.debye_length(I) * 1e9 for I in I_grid])       # nm
    dpsi = np.array([dl.biomolecule_potential(sigma_bio, d, I) for I in I_grid]) * 1e3
    return lam, dpsi


# ------------------------------------------------------------------- device
def device_vth_vs_ph(oxide_name, pH_points, ionic_strength, depth=1e-6):
    """Run the DDM.SPC ISFET transfer solve at each pH, return Vth(pH).

    The electrolyte surface potential psi_0(pH) enters the gate as an effective
    work-function shift dPhi = -psi_0 (a rise in surface potential raises the gate
    potential, i.e. lowers the effective work function).  Vth is extracted by the
    constant-current method from the I_D(V_G) transfer curve.
    """
    from DDM_SPC import read_gmsh, DriftDiffusionSolver

    mesh = read_gmsh("meshing/mosfet_v3.msh")
    ND, NA = 1e26, 1e23
    PHI0 = 4.10
    VD, VG = 0.10, np.round(np.arange(-0.6, 2.01, 0.1), 3)
    ox = dl.OXIDES[oxide_name]

    def transfer(phi_gate):
        s = DriftDiffusionSolver(
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
            recombination="srh+auger", device_depth=depth, verbose=False)
        s.solve_equilibrium()
        for vd in np.arange(0.02, VD + 1e-9, 0.02):
            s.solve_bias({"gate": 0.0, "drain": float(vd)})
        for vg in np.arange(-0.1, -0.6 - 1e-9, -0.1):
            s.solve_bias({"gate": float(vg), "drain": VD})
        ids = []
        for vg in VG:
            s.solve_bias({"gate": float(vg), "drain": VD})
            ids.append(abs(s.terminal_current("drain")))
        return np.array(ids)

    # constant-current threshold criterion from the first curve
    vths, psis = [], []
    i_th = None
    for pH in pH_points:
        psi0 = ox.surface_potential(pH, ionic_strength)
        psis.append(psi0)
        ids = transfer(PHI0 - psi0)                      # dPhi = -psi_0
        if i_th is None:
            i_th = 10 ** ((np.log10(ids.min()) + np.log10(ids.max())) / 2)
        vth = float("nan")
        for i in range(1, len(ids)):
            if ids[i - 1] < i_th <= ids[i]:
                l0, l1 = np.log10(ids[i - 1]), np.log10(ids[i])
                vth = VG[i - 1] + (np.log10(i_th) - l0) * (VG[i] - VG[i - 1]) / (l1 - l0)
                break
        vths.append(vth)
    return np.array(psis), np.array(vths)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--device", action="store_true",
                    help="also run the DDM.SPC device solve (slow)")
    args = ap.parse_args()

    oxides = ["Ta2O5", "Al2O3", "Si3N4", "SiO2"]
    pH_grid = np.arange(3.0, 11.01, 0.5)
    I_phys = 0.15                                        # physiological ionic strength

    print("ISFET / BioFET with electrolyte double layer (G-DL)")
    print("=" * 60)
    print(f"Nernst limit (ideal coupling) = {NERNST:.1f} mV/pH\n")

    resp = ph_response(oxides, pH_grid, I_phys)
    print(f"pH sensitivity at pH 7, I = {I_phys} M  (sub-Nernstian => coupling < 1):")
    print(f"  {'oxide':7}{'mV/pH':>9}{'alpha':>8}")
    i7 = int(np.argmin(abs(pH_grid - 7.0)))
    for name in oxides:
        s = abs(resp[name][1][i7])
        print(f"  {name:7}{s:9.1f}{s / NERNST:8.2f}")

    I_grid = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.15, 0.3])
    lam, dpsi = debye_screening(-0.01, 3.0, I_grid)
    print(f"\nBioFET Debye screening: bound charge -0.01 C/m^2 at d = 3 nm")
    print(f"  {'I (M)':>7}{'lambda_D (nm)':>15}{'response (mV)':>15}")
    for I, l, d in zip(I_grid, lam, dpsi):
        print(f"  {I:>7.3f}{l:>15.2f}{d:>15.2f}")
    print(f"  -> response falls ~{abs(dpsi[0] / dpsi[-2]):.0f}x from 1 mM to physiological"
          f" as lambda_D drops below d")

    if args.device:
        print("\n--- full device chain (DDM.SPC transfer-curve solve) ---")
        t0 = time.time()
        pH_pts = [4.0, 7.0, 10.0]
        for name in ["Ta2O5", "SiO2"]:
            psis, vths = device_vth_vs_ph(name, pH_pts, I_phys)
            # device Vth sensitivity from a linear fit of Vth vs pH
            slope = np.polyfit(pH_pts, vths, 1)[0] * 1e3          # mV/pH
            print(f"  {name:6}: Vth(pH) = "
                  + ", ".join(f"{p:.0f}:{v:+.3f}V" for p, v in zip(pH_pts, vths))
                  + f"   -> dVth/dpH = {abs(slope):.1f} mV/pH  (electrolyte-limited)")
        print(f"  device runs done in {time.time() - t0:.0f}s")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
        for name in oxides:
            a1.plot(pH_grid, resp[name][0] * 1e3, "-o", ms=3, label=name)
        a1.set_xlabel("bulk pH"); a1.set_ylabel("surface potential $\\psi_0$ (mV)")
        a1.set_title("Site-binding response (slope = sub-Nernstian sensitivity)")
        a1.grid(True); a1.legend()
        a2.semilogx(I_grid, dpsi, "-o")
        a2.set_xlabel("ionic strength (M)"); a2.set_ylabel("bound-charge response (mV)")
        a2.set_title("BioFET Debye-screening limit ($d$ = 3 nm)")
        a2.grid(True, which="both")
        fig.tight_layout(); fig.savefig("biosensor_isfet_gdl.png", dpi=130)
        print("\n  saved biosensor_isfet_gdl.png")


if __name__ == "__main__":
    main()
