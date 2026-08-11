"""Four-physics ablation for the GaAs/AlGaAs DH-LED paper (#8).

Runs the SAME 300nm x 10nm device under four progressively-richer physics
configurations and records I-V / IQE / wavelength, isolating each effect's
contribution:

  1. homo   : all-GaAs homojunction, Boltzmann, classical      (baseline)
  2. hetero : GaAs active / AlGaAs clad, Boltzmann, classical   (+ band offsets)
  3. +FD    : hetero, Fermi-Dirac, classical                    (+ degeneracy)
  4. +QM    : hetero, Fermi-Dirac, Schrodinger-Poisson          (full 4-physics)

Outputs JSON (numbers) + a 2-panel comparison figure into the scratchpad.
"""
import json
import time
import numpy as np

from DDM_SPC import DriftDiffusionSolver, RectangleMeshBuilder

LX, LY = 300e-9, 10e-9
X1, X2 = 100e-9, 200e-9


def build(hetero=True, stats="boltzmann", quantum=None, device_depth=1e-6):
    def region_of(xc, yc):
        if xc < X1:
            return "p_clad"
        if xc < X2:
            return "active"
        return "n_clad"

    mesh = RectangleMeshBuilder(LX, LY, nx=60, ny=16).build(
        region_of=region_of, region_names=["p_clad", "active", "n_clad"])

    clad_mat = "algaas" if hetero else "gaas"
    rp = {
        "active": {"material": "gaas",   "doping":  1e22},
        "p_clad": {"material": clad_mat, "doping": -1e24},
        "n_clad": {"material": clad_mat, "doping":  1e24},
    }
    return DriftDiffusionSolver(
        mesh=mesh,
        region_properties=rp,
        contacts={
            "anode":   {"type": "ohmic", "region": "p_clad", "voltage": 0.0,
                        "nodes": mesh.boundary_nodes["left"]},
            "cathode": {"type": "ohmic", "region": "n_clad", "voltage": 0.0,
                        "nodes": mesh.boundary_nodes["right"]},
        },
        recombination="srh+auger+radiative",
        quantum=quantum,
        statistics=stats,
        device_depth=device_depth,
        verbose=False,
    )


CONFIGS = [
    ("homo",   dict(hetero=False, stats="boltzmann",   quantum=None)),
    ("hetero", dict(hetero=True,  stats="boltzmann",   quantum=None)),
    ("+FD",    dict(hetero=True,  stats="fermi-dirac", quantum=None)),
    ("+QM",    dict(hetero=True,  stats="fermi-dirac",
                    quantum={"model": "schrodinger", "confinement": "y", "num_states": 4})),
]

Vgrid = np.round(np.arange(0.1, 1.61, 0.1), 3)


def run_ablation():
  results = {}
  for name, kw in CONFIGS:
    t0 = time.perf_counter()
    solver = build(**kw)
    sol = solver.solve_equilibrium()
    Vbi = float(sol.potential.max() - sol.potential.min())
    subbands = (np.round(solver.quantum.energies_n[:4], 4).tolist()
                if kw["quantum"] else None)
    rows = []
    for V in Vgrid:
        try:
            solver.solve_bias({"anode": float(V)})
            I = float(solver.terminal_current("anode"))
            opt = solver.optical()
            rows.append([float(V), I, float(opt.radiative_current),
                         float(opt.iqe), float(opt.wavelength_nm)])
        except Exception as e:  # noqa
            rows.append([float(V), None, None, None, None])
    peak_iqe = max((r[3] for r in rows if r[3] is not None), default=None)
    lam = next((r[4] for r in reversed(rows) if r[4] is not None), None)
    results[name] = dict(Vbi=Vbi, subbands=subbands, rows=rows,
                         peak_iqe=peak_iqe, wavelength_nm=lam,
                         runtime_s=round(time.perf_counter() - t0, 1))
    print(f"{name:8s} Vbi={Vbi:.4f}V  peak_IQE={peak_iqe}  lam={lam}  "
          f"({results[name]['runtime_s']}s)")

  out = "./ablation.json"
  with open(out, "w") as f:
      json.dump(results, f, indent=2)
  print("wrote", out)
  return results


if __name__ == "__main__":
    run_ablation()
