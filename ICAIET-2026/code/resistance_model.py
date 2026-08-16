"""
resistance_model.py
-------------------
Semi-empirical calm-water resistance model used as the *ground-truth* generator
for the machine-learning surrogate study reported in the ICAIET-2027 paper
"Data-Driven Surrogate Modelling for Early-Stage Ship-Hull Resistance
Prediction and Multi-Objective Design Exploration".

The model is a deliberately transparent reduced-order formulation that couples
two physically distinct components of the calm-water resistance of a
displacement ship:

  R_T = (1 + k) * R_F  +  R_R

  * R_F  -- frictional resistance from the ITTC-1957 model-ship correlation
            line, C_F = 0.075 / (log10(Re) - 2)^2.
  * (1+k) -- a form factor that accounts for the three-dimensional character of
             the boundary layer, correlated here with the block coefficient.
  * R_R  -- residuary (mainly wave-making) resistance, expressed through a
            non-dimensional coefficient C_R that grows steeply with Froude
            number and fullness, reproducing the well-known "resistance hump".

The wetted-surface and form-factor correlations are of the same algebraic
family as those used in classical preliminary-design methods (Holtrop-Mennen,
Watanabe). Numerical constants have been chosen so that predicted powers fall
in the correct order of magnitude for commercial displacement hulls; the model
is NOT intended as a substitute for tank testing or CFD, but as a controlled,
reproducible oracle whose functional complexity (non-linear, multiplicative,
regime-dependent) is representative of the mapping a surrogate must learn.

All quantities are SI unless stated otherwise.
"""

import numpy as np

# ---- physical constants (sea water, 15 degrees C) -------------------------
RHO = 1025.0          # density [kg/m^3]
NU = 1.19e-6          # kinematic viscosity [m^2/s]
G = 9.81              # gravitational acceleration [m/s^2]

# realistic design-space bounds for small-to-midsize commercial displacement
# hulls (feeder container / general cargo / tanker range).
BOUNDS = {
    "Lwl": (60.0, 230.0),   # waterline length [m]
    "B":   (10.0, 34.0),    # beam [m]
    "T":   (3.5, 13.0),     # draft [m]
    "Cb":  (0.55, 0.85),    # block coefficient [-]
    "V_kn": (10.0, 24.0),   # ship speed [knots]
}


def wetted_surface(Lwl, B, T, Cb):
    """Denny-Mumford style wetted-surface approximation [m^2]."""
    return 1.025 * Lwl * (Cb * B + 1.7 * T)


def form_factor(Cb, B, T, Lwl):
    """(1 + k) form factor, monotonically increasing with fullness and B/T."""
    return 1.0 + 0.11 + 0.55 * Cb ** 2 + 0.06 * (B / T) * (T / Lwl) ** 0.5


def residuary_coefficient(Fn, Cb, LB, BT):
    """
    Non-dimensional residuary-resistance coefficient C_R (x1e3).

    Captures the steep, regime-dependent growth of wave-making resistance:
      * negligible at low Fn,
      * a pronounced hump near Fn ~ 0.30-0.35 that is amplified by fullness,
      * a penalty for full, beamy hulls (high Cb, high B/T),
      * a slenderness (L/B) benefit.
    """
    hump = np.exp(-((Fn - 0.32) / 0.075) ** 2)            # primary wave hump
    base = 0.28 + 6.5 * Fn ** 4.2                          # smooth power-law rise
    fullness = 1.0 + 2.4 * (Cb - 0.55) ** 1.3             # fuller -> more waves
    slender = (7.0 / LB) ** 0.55                           # slender -> less waves
    beam_draft = 1.0 + 0.05 * (BT - 2.5)                  # beamy -> more waves
    cr = (base + 2.1 * hump * fullness) * slender * beam_draft
    return np.maximum(cr, 0.05)                            # x1e3, dimensionless


def total_resistance(Lwl, B, T, Cb, V_kn):
    """
    Total calm-water resistance R_T [kN] and its breakdown.
    Accepts scalars or numpy arrays (broadcast).
    Returns a dict of arrays.
    """
    Lwl = np.asarray(Lwl, dtype=float)
    B = np.asarray(B, dtype=float)
    T = np.asarray(T, dtype=float)
    Cb = np.asarray(Cb, dtype=float)
    V = np.asarray(V_kn, dtype=float) * 0.514444          # knots -> m/s

    Re = V * Lwl / NU
    Fn = V / np.sqrt(G * Lwl)

    Cf = 0.075 / (np.log10(Re) - 2.0) ** 2                # ITTC-1957
    S = wetted_surface(Lwl, B, T, Cb)
    k1 = form_factor(Cb, B, T, Lwl)

    Rf = 0.5 * RHO * V ** 2 * S * Cf                      # [N]
    Rf_visc = k1 * Rf

    LB = Lwl / B
    BT = B / T
    Cr = residuary_coefficient(Fn, Cb, LB, BT) * 1e-3     # dimensionless
    Rr = 0.5 * RHO * V ** 2 * S * Cr                      # [N]

    Rt = (Rf_visc + Rr) / 1000.0                          # -> kN
    Pe = Rt * V                                           # effective power [kW]

    return {
        "R_T_kN": Rt,
        "P_E_kW": Pe,
        "R_F_kN": Rf_visc / 1000.0,
        "R_R_kN": Rr / 1000.0,
        "Fn": Fn,
        "Re": Re,
        "S_m2": S,
    }


if __name__ == "__main__":
    # sanity check: a ~150 m feeder at 18 kn
    out = total_resistance(150.0, 24.0, 8.5, 0.68, 18.0)
    for k, v in out.items():
        print(f"{k:10s}: {float(v):12.3f}")
