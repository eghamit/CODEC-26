"""Validation suite: full FEM vs. compact model vs. closed-form physics.

Run:  python -m pytest test_models.py -q
Slow FEM-vs-compact checks are marked and use coarse meshes to stay fast.
"""

import numpy as np
import pytest

from fem_reference import DiskMembraneFEM, EPS0
from compact_model import CompactMicModel, _a_weight
import squeeze_film as sf

# reference design: 1 mm-radius Si3N4 diaphragm, 6 um gap
A_R, T, RHO_S, GAP, BIAS = 1.0e-3, 6.0, 3100 * 0.5e-6, 6.0e-6, 10.0


def fem(nr=16, nt=32):
    return DiskMembraneFEM(A_R, T, RHO_S, GAP, n_radial=nr, n_theta=nt)


def compact(**kw):
    return CompactMicModel.circular(A_R, T, RHO_S, GAP, bias=BIAS, **kw)


# ------------------------------------------------ FEM vs analytic membrane
def test_fem_center_deflection():
    f = fem()
    w = f.solve_static(0.0, 1.0)
    assert np.isclose(w[0], 1.0 * A_R**2 / (4 * T), rtol=2e-2)


def test_fem_fundamental_resonance():
    f = fem(20, 40)
    f01 = 2.4048 / (2 * np.pi * A_R) * np.sqrt(T / RHO_S)
    assert np.isclose(f.resonances(1)[0], f01, rtol=3e-2)


# ------------------------------------------------ compact closed forms
def test_compact_params_closed_form():
    cm = compact()
    A = np.pi * A_R**2
    assert np.isclose(cm.m, RHO_S * A / 3)
    assert np.isclose(cm.k, 2 * np.pi * T)
    assert np.isclose(cm.A_eff, A / 2)
    assert np.isclose(cm.int_phi2, A / 3)


def test_capacitance_closed_form_limits():
    cm = compact()
    # C(0) -> parallel plate
    assert np.isclose(cm.capacitance(0.0), EPS0 * cm.area / GAP, rtol=1e-9)
    # derivative limits vs general integrals: C'(0)=eps A/(2 g^2), C''(0)=2 eps A/(3 g^3)
    assert np.isclose(cm.dC(0.0), EPS0 * cm.area / (2 * GAP**2), rtol=1e-6)
    assert np.isclose(cm.d2C(0.0), 2 * EPS0 * cm.area / (3 * GAP**3), rtol=1e-6)


def test_capacitance_derivatives_finite_difference():
    cm = compact()
    q, h = 1.5e-6, 1e-10
    assert np.isclose(cm.dC(q), (cm.capacitance(q + h) - cm.capacitance(q - h)) / (2 * h), rtol=1e-5)
    assert np.isclose(cm.d2C(q), (cm.dC(q + h) - cm.dC(q - h)) / (2 * h), rtol=1e-5)


def test_capacitance_matches_fem_at_deflection():
    f = fem()
    cm = compact()
    w = f.solve_static(0.0, 40.0)          # deflection well below the gap
    assert np.isclose(cm.capacitance(w.max()), f.capacitance(w), rtol=1e-2)


# ------------------------------------------------ compact vs FEM (core)
@pytest.mark.slow
def test_bare_sensitivity_matches_fem():
    f = fem(20, 40)
    cm = compact()
    s_fem = f.sensitivity(BIAS)
    s_cm = cm.sensitivity(include_cavity=False)
    assert np.isclose(s_cm, s_fem, rtol=2e-2)


@pytest.mark.slow
def test_pullin_matches_fem():
    f = fem(10, 20)                        # coarse: pull-in continuation is costly
    cm = compact()
    assert np.isclose(cm.pull_in_voltage(), f.pull_in_voltage(), rtol=5e-2)


def test_pullin_below_gap_and_positive():
    cm = compact()
    assert 0 < cm.bias_point() < GAP / 3
    assert cm.pull_in_voltage() > BIAS


# ------------------------------------------------ squeeze film (derived Q)
def test_skvor_attenuation_limits():
    # K -> 0 as beta -> 1 (fully open backplate cannot squeeze the film)
    assert sf.skvor_attenuation(0.999) < sf.skvor_attenuation(0.5)
    assert sf.skvor_attenuation(1.0) == pytest.approx(0.0, abs=1e-9)
    assert sf.skvor_attenuation(0.3) > 0


def test_squeeze_damping_gap_scaling():
    # damping scales as 1/g0^3
    b1 = sf.damping_per_area(2e-6, 15e-6, 5e-6)
    b2 = sf.damping_per_area(4e-6, 15e-6, 5e-6)
    assert np.isclose(b1 / b2, 8.0, rtol=1e-9)


def test_quality_factor_is_derived_and_positive():
    cm = compact()
    assert cm.c > 0
    assert cm.quality_factor > 0


# ------------------------------------------------ acoustics
def test_cavity_stiffness_and_highpass():
    cm = compact(cavity_volume=5e-9, vent_radius=3e-6, vent_length=30e-6)
    assert np.isclose(cm.k_cav, cm.A_eff**2 / cm.C_ab)
    assert np.isclose(cm.f_hp, 1 / (2 * np.pi * cm.R_av * cm.C_ab))
    # a smaller cavity (stiffer) lowers sensitivity
    s_big = compact(cavity_volume=50e-9).sensitivity()
    s_small = compact(cavity_volume=2e-9).sensitivity()
    assert s_small < s_big


def test_frequency_response_highpass_and_midband():
    cm = compact(cavity_volume=10e-9, vent_radius=3e-6, vent_length=30e-6)
    f = np.array([cm.f_hp / 50, 1e3, 4e3])
    H = np.abs(cm.frequency_response(f))
    assert H[0] < 0.1 * H[1]                 # rolled off well below f_hp
    # mid-band magnitude tracks the analytic mid-band sensitivity
    assert np.isclose(H[1], cm.sensitivity(), rtol=0.1)


# ------------------------------------------------ noise / SNR
def test_a_weight_unity_at_1khz():
    assert np.isclose(_a_weight(1000.0), 1.0, rtol=2e-2)


def test_noise_and_snr_reasonable():
    cm = compact(cavity_volume=10e-9)
    ein = cm.equivalent_input_noise()
    assert ein["pa_rms"] > 0
    assert 40 < cm.snr() < 90            # plausible MEMS-mic SNR band


# ------------------------------------------------ large signal
def test_ngspice_macromodel_matches_python():
    import shutil
    if shutil.which("ngspice") is None:
        pytest.skip("ngspice not installed")
    import cosim_ngspice as co
    cm = compact(cavity_volume=10e-9)
    f, spice, py = co.verify_open_circuit(cm)
    mask = f < 0.5 * cm.resonance
    err = np.max(np.abs(spice[mask] - py[mask]) / py[mask])
    assert err < 1e-4          # SPICE macromodel reproduces the compact model


def test_transient_tracks_small_signal():
    cm = compact(cavity_volume=10e-9)
    ft = 1000.0
    t = np.linspace(0, 5 / ft, 400)
    p_amp = 0.1
    v = cm.transient(t, p_amp * np.sin(2 * np.pi * ft * t))
    tail = v[t > 2 / ft]
    v_amp = 0.5 * (tail.max() - tail.min())
    assert np.isclose(v_amp, cm.sensitivity() * p_amp, rtol=0.15)
