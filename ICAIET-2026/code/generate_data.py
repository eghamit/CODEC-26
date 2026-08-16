"""
generate_data.py
----------------
Build the synthetic hull-resistance dataset used to train and test the
surrogate models. Principal particulars are drawn by Latin Hypercube Sampling
(LHS) over the design bounds declared in resistance_model.BOUNDS so that the
five-dimensional input space is covered uniformly and without clustering.

Physically implausible combinations (e.g. draft exceeding a reasonable fraction
of the beam, or displacement outside a sensible envelope) are filtered so the
learner is not asked to interpolate over non-realisable hulls.

A small amount of multiplicative Gaussian noise (sigma = 2%) is added to the
target to emulate the scatter of towing-tank / CFD data around a mean trend.

Outputs
    data/hull_resistance.csv   -- full labelled dataset
"""

import os
import numpy as np

from resistance_model import BOUNDS, total_resistance

SEED = 20270113
N_SAMPLES = 6000
NOISE_SIGMA = 0.02

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


def latin_hypercube(n, d, rng):
    """Standard LHS on the unit hypercube -> (n, d) array in [0,1)."""
    cut = np.linspace(0, 1, n + 1)
    u = rng.uniform(size=(n, d))
    a, b = cut[:n], cut[1:]
    pts = np.empty((n, d))
    for j in range(d):
        pts[:, j] = a + (b - a) * u[:, j]
        rng.shuffle(pts[:, j])
    return pts


def main():
    rng = np.random.default_rng(SEED)
    keys = ["Lwl", "B", "T", "Cb", "V_kn"]
    lo = np.array([BOUNDS[k][0] for k in keys])
    hi = np.array([BOUNDS[k][1] for k in keys])

    # oversample, then filter to a clean realisable envelope.
    raw = latin_hypercube(int(N_SAMPLES * 3.0), 5, rng)
    X = lo + raw * (hi - lo)
    Lwl, B, T, Cb, V = X.T

    LB = Lwl / B
    BT = B / T
    Fn = (V * 0.514444) / np.sqrt(9.81 * Lwl)

    mask = (
        (LB >= 4.5) & (LB <= 9.5) &       # realistic length/beam
        (BT >= 2.0) & (BT <= 4.2) &       # realistic beam/draft
        (Fn <= 0.42)                      # displacement (non-planing) regime
    )
    X = X[mask][:N_SAMPLES]
    Lwl, B, T, Cb, V = X.T

    out = total_resistance(Lwl, B, T, Cb, V)
    Rt = out["R_T_kN"]
    noise = rng.normal(1.0, NOISE_SIGMA, size=Rt.shape)
    Rt_noisy = Rt * noise

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "hull_resistance.csv")
    header = "Lwl,B,T,Cb,V_kn,L_B,B_T,Fn,R_F_kN,R_R_kN,R_T_kN,P_E_kW"
    cols = np.column_stack([
        Lwl, B, T, Cb, V,
        Lwl / B, B / T, out["Fn"],
        out["R_F_kN"], out["R_R_kN"], Rt_noisy, Rt_noisy * (V * 0.514444),
    ])
    np.savetxt(path, cols, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"wrote {cols.shape[0]} samples -> {path}")
    print(f"R_T range: {Rt_noisy.min():.1f} .. {Rt_noisy.max():.1f} kN")
    print(f"Fn   range: {out['Fn'].min():.3f} .. {out['Fn'].max():.3f}")


if __name__ == "__main__":
    main()
