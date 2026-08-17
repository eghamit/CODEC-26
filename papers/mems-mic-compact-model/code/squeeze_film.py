"""Squeeze-film damping of a perforated MEMS backplate (derived, not fitted).

The air trapped in the thin gap between the moving diaphragm and the fixed,
perforated backplate is squeezed as the diaphragm moves; the viscous flow toward
the acoustic holes dissipates energy and sets the mechanical damping (hence the
quality factor).  This module derives the damping coefficient from first
principles rather than treating ``Q`` as a free parameter.

Model (Skvor / annular-cell).  Each acoustic hole vents one circular cell of
radius ``r_c`` (set by the hole pitch, pi r_c^2 = area/hole) with a central hole
of radius ``r_0``.  In the incompressible, low-frequency (viscous) limit the
linearised Reynolds equation in the cell,

    (g0^3 / 12 mu) (1/r) d/dr( r dP/dr ) = u          (u = diaphragm velocity)

with ``P(r_0)=0`` (vented) and ``dP/dr(r_c)=0`` (cell symmetry) integrates to a
net pressure whose area integral gives the per-cell damping force ``F = -b u``:

    b_cell  = (3 pi mu / 2 g0^3) r_c^4 K(beta),   beta = r_0 / r_c
    K(beta) = 4 beta^2 - beta^4 - 4 ln(beta) - 3        (Skvor attenuation fn)

so the damping per unit diaphragm area is ``b_area = b_cell / (pi r_c^2)``.
Projected onto the diaphragm mode shape (weight ``integral phi^2``) it yields the
modal damping ``c`` used by the compact model.  ``K -> 0`` as the perforation
fraction ``beta -> 1`` (a fully open backplate cannot squeeze the film), and the
1/g0^3 scaling is the hallmark of squeeze-film flow.
"""

from __future__ import annotations

import numpy as np

MU_AIR = 1.85e-5   # dynamic viscosity of air at ~25 C (Pa.s)


def skvor_attenuation(beta):
    """Skvor attenuation function K(beta), beta = hole/cell radius (0<beta<1)."""
    beta = np.asarray(beta, float)
    return 4.0 * beta**2 - beta**4 - 4.0 * np.log(beta) - 3.0


def damping_per_area(gap, cell_radius, hole_radius, viscosity=MU_AIR):
    """Squeeze-film damping per unit diaphragm area  b_area (N.s/m^3)."""
    beta = hole_radius / cell_radius
    b_cell = (3.0 * np.pi * viscosity / (2.0 * gap**3)) * cell_radius**4 \
        * skvor_attenuation(beta)
    return b_cell / (np.pi * cell_radius**2)


def cell_radius_from_holes(area, n_holes):
    """Cell radius r_c from total area and hole count:  pi r_c^2 = area/N."""
    return np.sqrt(area / (np.pi * n_holes))


def modal_damping(gap, cell_radius, hole_radius, int_phi2, viscosity=MU_AIR):
    """Modal damping coefficient c (N.s/m) = b_area * integral(phi^2 dA)."""
    return damping_per_area(gap, cell_radius, hole_radius, viscosity) * int_phi2


def quality_factor(m, k, c):
    """Mechanical quality factor Q = sqrt(m k) / c."""
    return np.sqrt(m * k) / c
