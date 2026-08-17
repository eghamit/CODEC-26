"""Reduced-order compact model of a capacitive MEMS microphone.

This is the behavioural macromodel targeted by ``mems_mic.va`` (Verilog-A): a
single mechanical mode coupled to the electrical port and to a back-cavity/vent
acoustic network.  Every equation here has a one-to-one Verilog-A counterpart,
so the Python model is both the design tool and the golden reference for the
compiled device.

Mathematical foundation
------------------------
1. Modal (Galerkin) reduction.  Writing the diaphragm deflection as
   ``w(r,t) = q(t) phi(r)`` with the fundamental (uniform-pressure) shape phi
   (peak-normalised), projection of the tensioned-membrane PDE gives a 1-DOF
   oscillator

       m q'' + c q' + k q = A_eff p_ac + F_es(q,V),
       m = integral rho_s phi^2 dA,  k = integral T |grad phi|^2 dA,
       A_eff = integral phi dA.

   For a clamped *circular* membrane the static shape is exactly parabolic,
   ``phi = 1 - (r/a)^2``, giving the closed forms  m = rho_s A/3,  k = 2 pi T,
   A_eff = A/2,  integral phi^2 = A/3   (A = pi a^2).

2. Electrostatics.  With that shape the mode-integrated gap capacitance has an
   exact closed form

       C(q) = (eps A / q) ln( g0 / (g0 - q) ),        C(0) = eps A / g0,

   and the generalised electrostatic force is  F_es = 1/2 V^2 C'(q).  Pull-in is
   the turning point  q = C'/C''  ,  V_PI = sqrt( 2 k / C''(q_PI) ).

3. Transduction.  Linearising Q = C(q) V about the bias point gives the coupling
   factor  Gamma = V_B C'(q0)  and the electrostatic spring softening
   k_es = 1/2 V_B^2 C''(q0); the open-circuit (constant-charge) output is
   ``v_oc = (Gamma / C0) q``.

4. Acoustics.  A back cavity of volume ``V_b`` adds the acoustic compliance
   ``C_ab = V_b/(rho0 c0^2)`` (modal stiffening ``k_cav = A_eff^2 / C_ab``); a
   pressure-equalisation vent (resistance ``R_av``, mass ``M_av``) shorts the
   diaphragm at low frequency, setting the high-pass corner
   ``f_hp = 1/(2 pi R_av C_ab)``.

5. Damping and noise.  ``c`` comes from the derived squeeze-film model
   (``squeeze_film.py``); by fluctuation-dissipation it also fixes the
   thermal-mechanical input-referred noise ``S_p = 4 kB T c / A_eff^2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import squeeze_film as sf

EPS0 = 8.8541878128e-12
KB = 1.380649e-23
RHO0 = 1.204        # air density (kg/m^3, 20 C)
C0_SOUND = 343.0    # speed of sound (m/s)
P_REF = 20e-6       # 20 uPa reference (0 dB SPL)


@dataclass
class CompactMicModel:
    # --- lumped mechanical/electrical parameters ---
    m: float                 # modal mass (kg)
    k: float                 # modal stiffness (N/m)
    A_eff: float             # effective area (m^2)
    area: float              # full diaphragm area (m^2)
    int_phi2: float          # integral phi^2 dA (m^2)
    gap: float               # electrode gap (m)
    eps: float = EPS0
    bias: float = 10.0       # DC bias (V)
    temperature: float = 293.0
    # --- squeeze-film / backplate ---
    cell_radius: float = 20e-6
    hole_radius: float = 5e-6
    viscosity: float = sf.MU_AIR
    # --- acoustics ---
    cavity_volume: float = 1e-9     # back-cavity volume (m^3)
    vent_radius: float = 5e-6       # equalisation-vent radius (m)
    vent_length: float = 20e-6      # vent channel length (m)
    sealed_cavity: bool = False     # if True, cavity stiffens the DC pull-in too

    c: float = field(init=False)    # modal damping (N.s/m)

    def __post_init__(self):
        self.c = sf.modal_damping(self.gap, self.cell_radius, self.hole_radius,
                                  self.int_phi2, self.viscosity)

    # ---------------------------------------------------- constructors
    @classmethod
    def circular(cls, radius, tension, areal_density, gap, **kw):
        """Closed-form compact model of a clamped circular membrane."""
        A = np.pi * radius**2
        return cls(m=areal_density * A / 3.0, k=2.0 * np.pi * tension,
                   A_eff=A / 2.0, area=A, int_phi2=A / 3.0, gap=gap, **kw)

    @classmethod
    def from_fem(cls, fem, **kw):
        """Compact model with parameters extracted from a full-FEM solve."""
        mp = fem.modal_parameters()
        return cls(m=mp["m"], k=mp["k"], A_eff=mp["A_eff"], area=fem.area,
                   int_phi2=mp["int_phi2"], gap=fem.gap, eps=fem.eps, **kw)

    # ---------------------------------------------------- capacitance C(q)
    # Exact for the parabolic fundamental shape; series-expanded near q=0.
    def capacitance(self, q):
        A, g = self.area, self.gap
        x = q / g
        if abs(x) < 1e-6:
            return self.eps * A / g * (1 + x / 2 + x**2 / 3 + x**3 / 4)
        return self.eps * A / q * np.log(g / (g - q))

    def dC(self, q):
        A, g = self.area, self.gap
        x = q / g
        if abs(x) < 1e-4:
            return self.eps * A / g**2 * (0.5 + 2 * x / 3 + 3 * x**2 / 4)
        L = -np.log(1.0 - q / g)
        return self.eps * A * (q / (g - q) - L) / q**2

    def d2C(self, q):
        A, g = self.area, self.gap
        x = q / g
        if abs(x) < 1e-3:
            return self.eps * A / g**3 * (2.0 / 3 + 1.5 * x + 12 * x**2 / 5)
        L = -np.log(1.0 - q / g)
        u = q / (g - q) - L                       # = C' * q^2/(eps A)
        return self.eps * A * (q**2 / (g - q)**2 - 2 * u) / q**3

    # ---------------------------------------------------- acoustics
    @property
    def C_ab(self):
        return self.cavity_volume / (RHO0 * C0_SOUND**2)

    @property
    def k_cav(self):
        return self.A_eff**2 / self.C_ab

    @property
    def R_av(self):                                # Poiseuille vent resistance
        return 8.0 * self.viscosity * self.vent_length / (np.pi * self.vent_radius**4)

    @property
    def M_av(self):                                # vent acoustic mass
        return RHO0 * self.vent_length / (np.pi * self.vent_radius**2)

    @property
    def f_hp(self):
        return 1.0 / (2 * np.pi * self.R_av * self.C_ab)

    # ---------------------------------------------------- statics / pull-in
    def _k_dc(self):
        return self.k + (self.k_cav if self.sealed_cavity else 0.0)

    def bias_point(self, V=None):
        V = self.bias if V is None else V
        q, k_dc = 0.0, self._k_dc()
        for _ in range(100):
            g = k_dc * q - 0.5 * V**2 * self.dC(q)
            dg = k_dc - 0.5 * V**2 * self.d2C(q)
            if dg <= 0:
                raise RuntimeError("bias beyond pull-in")
            step = g / dg
            q -= step
            q = min(max(q, -0.999 * self.gap), 0.999 * self.gap)
            if abs(step) < 1e-16:
                break
        return q

    def pull_in_voltage(self):
        k_dc = self._k_dc()
        lo, hi = 1e-12, 0.999 * self.gap

        def h(q):
            return q - self.dC(q) / self.d2C(q)
        flo = h(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if flo * h(mid) <= 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-15:
                break
        q_pi = 0.5 * (lo + hi)
        return np.sqrt(2.0 * k_dc / self.d2C(q_pi))

    def k_es(self, V=None, q0=None):
        V = self.bias if V is None else V
        q0 = self.bias_point(V) if q0 is None else q0
        return 0.5 * V**2 * self.d2C(q0)

    # ---------------------------------------------------- small signal
    def sensitivity(self, V=None, include_cavity=True):
        """Mid-band open-circuit sensitivity |dVout/dp| (V/Pa).

        Uses the dynamic stiffness k + k_cav - k_es (above f_hp, below f0).
        Set ``include_cavity=False`` for the bare-diaphragm value that the
        cavity-free full-FEM reference computes (used for core verification).
        """
        V = self.bias if V is None else V
        q0 = self.bias_point(V)
        C0 = self.capacitance(q0)
        Gamma = V * self.dC(q0)
        k_cav = self.k_cav if include_cavity else 0.0
        k_net = self.k + k_cav - self.k_es(V, q0)
        return abs((Gamma / C0) * self.A_eff / k_net)

    def frequency_response(self, freqs, V=None):
        """Complex output sensitivity Vout/p (V/Pa) vs frequency (Hz).

        Solves the coupled diaphragm + back-cavity + vent 2-node acoustic
        network per frequency, so the response shows the vent high-pass, the
        cavity stiffening and the mechanical resonance together.
        """
        V = self.bias if V is None else V
        q0 = self.bias_point(V)
        C0 = self.capacitance(q0)
        Gamma = V * self.dC(q0)
        k_net = self.k - self.k_es(V, q0)          # cavity enters via the network
        w = 2 * np.pi * np.asarray(freqs, float)
        jw = 1j * w
        Zm = k_net - w**2 * self.m + jw * self.c
        Zvent = self.R_av + jw * self.M_av
        G = jw * self.A_eff**2 / Zm + 1.0 / Zvent
        p_b = G / (G + jw * self.C_ab)             # cavity pressure / p_in
        q = self.A_eff * (1.0 - p_b) / Zm          # q / p_in
        return (Gamma / C0) * q

    def transducer_tf_midband(self, freqs, V=None):
        """Bare 1-DOF transducer TF Vout/p (V/Pa), cavity folded into stiffness,
        no vent.  This is exactly the linear small-signal network realised in
        the SPICE/Verilog-A macromodel, used to cross-check the compiled device.
        """
        V = self.bias if V is None else V
        q0 = self.bias_point(V)
        C0 = self.capacitance(q0)
        Gamma = V * self.dC(q0)
        k_net = self.k + self.k_cav - self.k_es(V, q0)
        w = 2 * np.pi * np.asarray(freqs, float)
        Zm = k_net - w**2 * self.m + 1j * w * self.c
        return (Gamma / C0) * self.A_eff / Zm

    def spice_parameters(self, V=None):
        """Small-signal macromodel parameters for the SPICE/Verilog-A netlist."""
        V = self.bias if V is None else V
        q0 = self.bias_point(V)
        return dict(M=self.m, Cdamp=self.c,
                    Knet=self.k + self.k_cav - self.k_es(V, q0),
                    Aeff=self.A_eff, Gamma=V * self.dC(q0),
                    C0=self.capacitance(q0), bias=V)

    @property
    def resonance(self):
        q0 = self.bias_point()
        k_net = self.k + self.k_cav - self.k_es(q0=q0)
        return np.sqrt(max(k_net, 0.0) / self.m) / (2 * np.pi)

    @property
    def quality_factor(self):
        k_net = self.k + self.k_cav
        return np.sqrt(self.m * k_net) / self.c

    # ---------------------------------------------------- noise / SNR
    def input_noise_psd(self):
        """Thermal-mechanical input-referred pressure PSD S_p (Pa^2/Hz), white."""
        S_F = 4.0 * KB * self.temperature * self.c        # force PSD (N^2/Hz)
        return S_F / self.A_eff**2

    def equivalent_input_noise(self, f_lo=20.0, f_hi=20e3, a_weight=True):
        """Integrated equivalent input noise -> (Pa_rms, dBSPL, dBA)."""
        Sp = self.input_noise_psd()
        f = np.linspace(f_lo, f_hi, 4000)
        # signal and noise pass through the same mechanical TF, so the
        # input-referred PSD is flat; A-weighting shapes only the perceptual band
        w_un = np.ones_like(f)
        w_a = _a_weight(f) if a_weight else w_un
        p_un = np.sqrt(np.trapezoid(Sp * w_un, f))
        p_a = np.sqrt(np.trapezoid(Sp * w_a**2, f))
        dbspl = 20 * np.log10(p_un / P_REF)
        dba = 20 * np.log10(p_a / P_REF)
        return dict(pa_rms=p_un, dbspl=dbspl, dba=dba)

    def snr(self, f_lo=20.0, f_hi=20e3):
        """A-weighted SNR (dB) referenced to 94 dB SPL (1 Pa), per IEC 61672."""
        ein = self.equivalent_input_noise(f_lo, f_hi, a_weight=True)
        return 94.0 - ein["dba"]

    # ---------------------------------------------------- large signal
    def transient(self, t, pressure, V=None):
        """Nonlinear constant-charge transient -> output voltage (V), for THD.

        Integrates  m q'' + c q' + (k+k_cav) q = A_eff p(t) + Q0^2/(2 C^2) C'(q)
        about the bias point (mid-band; vent/cavity as static stiffening).
        """
        from scipy.integrate import solve_ivp
        from scipy.interpolate import interp1d
        V = self.bias if V is None else V
        t = np.asarray(t, float)
        p_of_t = (pressure if callable(pressure)
                  else interp1d(t, np.asarray(pressure, float), kind="linear",
                                bounds_error=False, fill_value=0.0))
        q0 = self.bias_point(V)
        Q0 = self.capacitance(q0) * V
        k_tot = self.k + self.k_cav
        qmax = 0.999 * self.gap

        def rhs(tt, y):
            q, qd = y
            q = min(max(q, -qmax), qmax)
            C = self.capacitance(q)
            f_es = Q0**2 / (2 * C**2) * self.dC(q)
            acc = (self.A_eff * float(p_of_t(tt)) + f_es
                   - self.c * qd - k_tot * q) / self.m
            return [qd, acc]

        sol = solve_ivp(rhs, (t[0], t[-1]), [q0, 0.0], t_eval=t, method="RK45",
                        rtol=1e-9, atol=1e-15, max_step=(t[1] - t[0]))
        q = np.clip(sol.y[0], -qmax, qmax)
        Cq = np.array([self.capacitance(qi) for qi in q])
        return Q0 / Cq - V

    # ---------------------------------------------------- report
    def summary(self):
        q0 = self.bias_point()
        return {
            "modal_mass_kg": self.m,
            "modal_stiffness_N_per_m": self.k,
            "cavity_stiffness_N_per_m": self.k_cav,
            "effective_area_m2": self.A_eff,
            "squeeze_damping_Ns_per_m": self.c,
            "quality_factor": self.quality_factor,
            "rest_capacitance_F": self.capacitance(0.0),
            "bias_V": self.bias,
            "pull_in_V": self.pull_in_voltage(),
            "bias_deflection_m": q0,
            "resonance_Hz": self.resonance,
            "highpass_corner_Hz": self.f_hp,
            "sensitivity_mV_per_Pa": self.sensitivity() * 1e3,
            "sensitivity_dBV": 20 * np.log10(self.sensitivity()),
            "EIN_dBA": self.equivalent_input_noise()["dba"],
            "SNR_dBA": self.snr(),
        }


def _a_weight(f):
    """IEC 61672 A-weighting amplitude factor (linear, normalised to 1 kHz)."""
    f = np.asarray(f, float)
    f2 = f**2
    ra = (12194.0**2 * f2**2) / (
        (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
        * (f2 + 12194.0**2))
    return ra / 0.7943282347  # +2.0 dB normalisation at 1 kHz
