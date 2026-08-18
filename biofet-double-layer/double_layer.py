"""Electrolyte double-layer + site-binding gate model for ISFET / BioFET (G-DL).

Turns an analyte state (bulk pH, ionic strength, and any bound-biomolecule
surface charge) into the **effective surface potential** psi_0 that reaches the
transistor gate.  This is the physics the bare device solver lacks: it is what
makes the pH response *sub-Nernstian* (< 59.5 mV/pH) and makes biomolecule
detection *fail beyond the Debye length* -- the two effects that separate a real
BioFET from an ideal MOS threshold shift.

Model (standard, citable):
  * Site-binding (2-pK amphoteric) surface charge sigma_0(psi_0), with the
    surface proton activity Boltzmann-shifted by psi_0
    (Yates-Levine-Healy / Fung-Ko / van Hal-Bergveld 1996).
  * Gouy-Chapman-Stern double layer: a Stern (Helmholtz) capacitance in series
    with the diffuse layer, sigma_d(psi_d) = -sqrt(8 eps kT n0) sinh(q psi_d/2kT).
  * Charge balance sigma_0 = -sigma_d, with psi_0 - psi_d = sigma_0 / C_stern,
    solved self-consistently for psi_d (hence psi_0) at each pH.
  * Biomolecule charge sigma_bio at distance d: linearised Debye-Huckel screening
    delta_psi = (sigma_bio/(eps eps0 kappa)) exp(-kappa d),  kappa = 1/lambda_D.

Units: SI throughout (V, C, m, F/m^2); pH and pK dimensionless.
"""
import numpy as np
from scipy.optimize import brentq

# --- physical constants -------------------------------------------------------
Q = 1.602176634e-19        # C
KB = 1.380649e-23          # J/K
EPS0 = 8.8541878128e-12    # F/m
NA = 6.02214076e23         # 1/mol
EPS_WATER = 78.5           # relative permittivity of water (~298 K)


def thermal_voltage(T=298.15):
    return KB * T / Q


def debye_length(ionic_strength_M, T=298.15, eps_r=EPS_WATER):
    """Debye screening length [m] for a symmetric 1:1 electrolyte.

    ionic_strength_M in mol/L.  ~0.30 nm / sqrt(I[M]) at room temperature.
    """
    I = ionic_strength_M * 1e3 * NA          # ions/m^3 per species (mol/m^3 -> 1/m^3)
    return np.sqrt(eps_r * EPS0 * KB * T / (2.0 * Q * Q * I))


def _sigma_diffuse(psi_d, ionic_strength_M, T=298.15, eps_r=EPS_WATER):
    """Gouy-Chapman diffuse-layer charge density [C/m^2] at plane potential psi_d."""
    n0 = ionic_strength_M * 1e3 * NA
    pref = np.sqrt(8.0 * eps_r * EPS0 * KB * T * n0)
    arg = np.clip(Q * psi_d / (2.0 * KB * T), -60.0, 60.0)   # guard sinh overflow
    return -pref * np.sinh(arg)


class OxideSurface:
    """A functionalised gate-oxide surface described by a 2-pK site-binding model.

    Parameters (representative 300 K literature values):
      N_s      site density [1/m^2]
      pKa, pKb acid/base dissociation constants (pzc = (pKa+pKb)/2)
      C_stern  Stern (Helmholtz) capacitance [F/m^2]
    """

    def __init__(self, name, N_s, pKa, pKb, C_stern=0.20):
        self.name = name
        self.N_s = float(N_s)
        self.pKa = float(pKa)
        self.pKb = float(pKb)
        self.C_stern = float(C_stern)

    @property
    def pzc(self):
        return 0.5 * (self.pKa + self.pKb)

    def sigma_site(self, psi_0, pH_bulk, T=298.15):
        """Site-binding surface charge [C/m^2] at surface potential psi_0."""
        Ka, Kb = 10.0 ** (-self.pKa), 10.0 ** (-self.pKb)
        # surface proton activity: h_s = h_bulk * exp(-q psi_0 / kT)
        h_bulk = 10.0 ** (-pH_bulk)
        arg = np.clip(-Q * psi_0 / (KB * T), -60.0, 60.0)
        h = h_bulk * np.exp(arg)
        # amphoteric fractions relative to neutral MOH
        r_plus = h / Ka                      # [MOH2+]/[MOH]
        r_minus = Kb / h                     # [MO-]/[MOH]
        denom = 1.0 + r_plus + r_minus
        theta = (r_plus - r_minus) / denom   # net protonation per site
        return Q * self.N_s * theta

    def surface_potential(self, pH_bulk, ionic_strength_M, T=298.15,
                          eps_r=EPS_WATER):
        """Self-consistent surface potential psi_0 [V] at (pH, ionic strength)."""
        # Root-find on psi_0: the site charge is bounded (|sigma_0| <= q N_s), so
        # psi_0 stays in a physical window and the diffuse-plane potential
        # psi_d = psi_0 - sigma_0/C_stern follows. Charge balance:
        #   sigma_site(psi_0) + sigma_diffuse(psi_d) = 0.
        def residual(psi_0):
            sigma_0 = self.sigma_site(psi_0, pH_bulk, T)
            psi_d = psi_0 - sigma_0 / self.C_stern
            return sigma_0 + _sigma_diffuse(psi_d, ionic_strength_M, T, eps_r)

        lo, hi = -0.6, 0.6
        return brentq(residual, lo, hi, xtol=1e-10, rtol=1e-12)

    def ph_sensitivity(self, pH0, ionic_strength_M, dpH=0.05, T=298.15):
        """d(psi_0)/d(pH) [V/pH] about pH0 (central difference)."""
        pa = self.surface_potential(pH0 + dpH, ionic_strength_M, T)
        pb = self.surface_potential(pH0 - dpH, ionic_strength_M, T)
        return (pa - pb) / (2.0 * dpH)


def biomolecule_potential(sigma_bio, distance_m, ionic_strength_M,
                          T=298.15, eps_r=EPS_WATER):
    """Debye-screened surface potential shift [V] from a bound charge sheet.

    Linearised Debye-Huckel: a charge plane of density sigma_bio [C/m^2] a
    distance ``distance_m`` from the sensor surface raises the surface potential
    by (sigma_bio / (eps eps0 kappa)) exp(-kappa d).  The exp(-d/lambda_D) factor
    is the Debye-screening limit that kills BioFET response at high ionic strength.
    """
    lam = debye_length(ionic_strength_M, T, eps_r)
    kappa = 1.0 / lam
    return sigma_bio / (eps_r * EPS0 * kappa) * np.exp(-distance_m * kappa)


# Representative functionalised gate oxides (site density in 1/m^2; 1e18/m^2 =
# 1e14/cm^2).  The (N_s, pKa, pKb) triples are *illustrative* values chosen so the
# self-consistent site-binding solve reproduces the well-known ISFET ordering and
# pH-sensitivity ranges (Ta2O5, Al2O3 ~ near-Nernstian; Si3N4 intermediate; SiO2
# strongly sub-Nernstian); they are meant to be re-fit to a given fabrication, not
# taken as material constants.  Buffer capacity rises with N_s and falls with the
# pK spread DpK = pKb - pKa, which is what sets the sensitivity parameter alpha.
OXIDES = {
    # name       N_s        pKa   pKb   C_stern
    "SiO2":  OxideSurface("SiO2",  2.0e17, -1.0, 7.0, 0.20),   # ~30 mV/pH
    "Si3N4": OxideSurface("Si3N4", 4.0e18,  3.0, 7.0, 0.30),   # ~50 mV/pH
    "Al2O3": OxideSurface("Al2O3", 8.0e18,  6.5, 10.5, 0.30),  # ~55 mV/pH
    "Ta2O5": OxideSurface("Ta2O5", 1.0e19,  2.0, 4.0, 0.40),   # ~58 mV/pH
}
