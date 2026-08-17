"""Verification study: compact model vs. full FEM (accuracy + speed).

Runs three diaphragm designs through both the high-fidelity FEM reference and
the reduced-order compact model, records the electro-mechanical figures of merit
and the wall-clock cost of each, and writes ``data/results.json`` consumed by
``make_figures.py`` and the paper.

The comparison is bare-diaphragm (no acoustic cavity) so both models solve the
*same* physics -- this isolates the error introduced by the modal reduction
itself.  The compact model then adds the cavity/vent/noise physics that FEM
alone does not provide.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from fem_reference import DiskMembraneFEM
from compact_model import CompactMicModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

# (label, radius[m], tension[N/m], thickness[m], gap[m], bias[V])
DESIGNS = [
    ("D1  0.8mm / T4",  0.8e-3, 4.0, 0.5e-6, 5e-6, 8.0),
    ("D2  1.0mm / T6",  1.0e-3, 6.0, 0.5e-6, 6e-6, 10.0),
    ("D3  1.2mm / T8",  1.2e-3, 8.0, 0.5e-6, 7e-6, 12.0),
]
RHO = 3100.0


def run(nr=18, nt=36, do_pullin=True):
    rows = []
    for label, a, T, th, gap, bias in DESIGNS:
        rho_s = RHO * th
        # ---- compact (closed form) ----
        cm = CompactMicModel.circular(a, T, rho_s, gap, bias=bias)
        t0 = time.perf_counter()
        s_cm = cm.sensitivity(include_cavity=False)
        vpi_cm = cm.pull_in_voltage()
        f0_cm = np.sqrt(cm.k / cm.m) / (2 * np.pi)
        c0_cm = cm.capacitance(0.0)
        t_cm = time.perf_counter() - t0

        # ---- full FEM ----
        fem = DiskMembraneFEM(a, T, rho_s, gap, n_radial=nr, n_theta=nt)
        t0 = time.perf_counter()
        s_fem = fem.sensitivity(bias)
        t_sens = time.perf_counter() - t0
        f0_fem = float(fem.resonances(1)[0])
        c0_fem = fem.capacitance(np.zeros(fem.nn))
        if do_pullin:
            t0 = time.perf_counter()
            vpi_fem = fem.pull_in_voltage()
            t_pi = time.perf_counter() - t0
        else:
            vpi_fem, t_pi = None, None

        rows.append(dict(
            label=label, radius=a, tension=T, gap=gap, bias=bias,
            nodes=int(fem.nn), elements=int(len(fem.tris)),
            compact=dict(sensitivity=s_cm, pull_in=vpi_cm, f0=f0_cm,
                         C0=c0_cm, time_s=t_cm),
            fem=dict(sensitivity=s_fem, pull_in=vpi_fem, f0=f0_fem,
                     C0=c0_fem, time_sensitivity_s=t_sens, time_pullin_s=t_pi),
            error=dict(
                sensitivity=abs(s_cm - s_fem) / s_fem,
                f0=abs(f0_cm - f0_fem) / f0_fem,
                C0=abs(c0_cm - c0_fem) / c0_fem,
                pull_in=(abs(vpi_cm - vpi_fem) / vpi_fem
                         if vpi_fem else None)),
            speedup=dict(
                sensitivity=t_sens / t_cm,
                pullin=(t_pi / t_cm if t_pi else None)),
        ))
        print(f"{label}: sens {s_cm*1e3:.2f}/{s_fem*1e3:.2f} mV/Pa "
              f"(err {rows[-1]['error']['sensitivity']*100:.1f}%), "
              f"f0 {f0_cm/1e3:.1f}/{f0_fem/1e3:.1f} kHz, "
              f"Vpi {vpi_cm:.2f}/{vpi_fem if vpi_fem else float('nan'):.2f} V, "
              f"FEM {t_sens+ (t_pi or 0):.1f}s vs compact {t_cm*1e3:.2f}ms")
    return rows


def main():
    os.makedirs(DATA, exist_ok=True)
    rows = run()
    # headline: reference operating point (D2) full compact summary + acoustics
    a, T, th, gap, bias = 1.0e-3, 6.0, 0.5e-6, 6e-6, 10.0
    ref = CompactMicModel.circular(a, T, RHO * th, gap, bias=bias,
                                   cavity_volume=10e-9, vent_radius=3e-6,
                                   vent_length=30e-6, cell_radius=15e-6,
                                   hole_radius=5e-6)
    out = dict(designs=rows, reference_summary=ref.summary())
    with open(os.path.join(DATA, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(DATA, 'results.json')}")


if __name__ == "__main__":
    main()
