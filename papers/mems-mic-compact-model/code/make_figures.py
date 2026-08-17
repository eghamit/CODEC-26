"""Generate the publication figures for the MEMS-mic compact-model paper.

Reads ../data/results.json (full-FEM numbers) and recomputes the fast
compact-model / ngspice curves live.  Writes ../figures/*.png.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compact_model import CompactMicModel
from fem_reference import DiskMembraneFEM
import squeeze_film as sf
import cosim_ngspice as co

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130, "lines.linewidth": 1.8})
C_FEM, C_CM, C_SF, C_CA = "#1f77b4", "#d62728", "#2ca02c", "#9467bd"

RHO = 3100.0
REF = dict(radius=1.0e-3, tension=6.0, thick=0.5e-6, gap=6e-6, bias=10.0,
           cavity_volume=10e-9, vent_radius=3e-6, vent_length=30e-6,
           cell_radius=15e-6, hole_radius=5e-6)


def ref_model(**over):
    p = dict(REF); p.update(over)
    return CompactMicModel.circular(
        p["radius"], p["tension"], RHO * p["thick"], p["gap"], bias=p["bias"],
        cavity_volume=p["cavity_volume"], vent_radius=p["vent_radius"],
        vent_length=p["vent_length"], cell_radius=p["cell_radius"],
        hole_radius=p["hole_radius"])


def results():
    with open(os.path.join(DATA, "results.json")) as f:
        return json.load(f)


# --------------------------------------------------- fig 1: FEM vs compact
def fig_validation():
    r = results()["designs"]
    labels = [d["label"].split()[0] for d in r]
    x = np.arange(len(r))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, (key, title, unit, scale) in enumerate([
            ("sensitivity", "Sensitivity", "mV/Pa", 1e3),
            ("f0", "Resonance", "kHz", 1e-3),
            ("pull_in", "Pull-in voltage", "V", 1.0)]):
        cm = [d["compact"][key] * scale for d in r]
        fe = [d["fem"][key] * scale for d in r]
        ax[i].bar(x - 0.2, fe, 0.4, label="full FEM", color=C_FEM)
        ax[i].bar(x + 0.2, cm, 0.4, label="compact", color=C_CM)
        for j, d in enumerate(r):
            e = d["error"][key]
            ax[i].text(x[j], max(cm[j], fe[j]) * 1.02, f"{e*100:.1f}%",
                       ha="center", fontsize=8)
        ax[i].set_xticks(x); ax[i].set_xticklabels(labels)
        ax[i].set_title(title); ax[i].set_ylabel(unit)
        ax[i].set_ylim(0, max(max(cm), max(fe)) * 1.15)
    ax[0].legend(fontsize=9)
    fig.suptitle("Compact model vs. full FEM (label = relative error)", y=1.02)
    fig.tight_layout(); _save(fig, "fig1_validation")


# --------------------------------------------------- fig 2: frequency response
def fig_frequency_response():
    cm = ref_model()
    f = np.logspace(1, np.log10(1.2e5), 400)
    H = np.abs(cm.frequency_response(f))
    fsf, Hsf = co.source_follower(cm, f_lo=10, f_hi=1.2e5)
    fca, Hca = co.charge_amplifier(cm, f_lo=10, f_hi=1.2e5)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    db = lambda y: 20 * np.log10(np.clip(y, 1e-9, None))
    ax.semilogx(f, db(H), color=C_CM, label="compact transducer (open-circuit)")
    ax.semilogx(fsf, db(Hsf), color=C_SF, label="+ NMOS source follower (ngspice)")
    ax.semilogx(fca, db(Hca), color=C_CA, label="+ op-amp charge amp (ngspice)")
    ax.axvline(cm.f_hp, ls=":", color="grey")
    ax.text(cm.f_hp * 1.2, ax.get_ylim()[0] + 4, f"vent $f_{{hp}}$={cm.f_hp:.2g} Hz",
            fontsize=8, color="grey")
    ax.axvline(cm.resonance, ls=":", color="k")
    ax.text(cm.resonance * 0.4, -35, f"$f_0$={cm.resonance/1e3:.0f} kHz", fontsize=8)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("|Vout/p| (dBV re 1 V/Pa)")
    ax.set_title("Transducer + CMOS read-out co-simulation")
    ax.legend(fontsize=8.5, loc="lower center"); ax.set_ylim(-70, 10)
    fig.tight_layout(); _save(fig, "fig2_frequency_response")


# --------------------------------------------------- fig 3: speed vs accuracy
def fig_speed_accuracy():
    r = results()["designs"]
    labels = [d["label"].split()[0] for d in r]
    x = np.arange(len(r))
    t_fem = [d["fem"]["time_sensitivity_s"] + (d["fem"]["time_pullin_s"] or 0)
             for d in r]
    t_cm = [d["compact"]["time_s"] for d in r]
    speed = [tf / tc for tf, tc in zip(t_fem, t_cm)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(x - 0.2, t_fem, 0.4, label="full FEM", color=C_FEM)
    ax[0].bar(x + 0.2, t_cm, 0.4, label="compact", color=C_CM)
    ax[0].set_yscale("log"); ax[0].set_ylabel("evaluation time (s)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].legend(fontsize=9)
    ax[0].set_title("Cost per design evaluation")
    ax[1].bar(x, speed, 0.5, color="#555")
    for j, s in enumerate(speed):
        ax[1].text(x[j], s, f"{s:.0e}", ha="center", va="bottom", fontsize=9)
    ax[1].set_yscale("log"); ax[1].set_ylabel("compact speed-up (x)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    errs = [d["error"]["sensitivity"] * 100 for d in r]
    ax[1].set_title(f"Speed-up (sensitivity error < {max(errs):.1f}%)")
    fig.tight_layout(); _save(fig, "fig3_speed_accuracy")


# --------------------------------------------------- fig 4: derived squeeze Q
def fig_squeeze_film():
    cm = ref_model()
    gaps = np.linspace(2e-6, 12e-6, 60)
    Q_gap = [ref_model(gap=g).quality_factor for g in gaps]
    beta = np.linspace(0.05, 0.9, 60)
    rc = cm.cell_radius
    Q_beta = []
    for b in beta:
        Q_beta.append(ref_model(hole_radius=b * rc).quality_factor)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(gaps * 1e6, Q_gap, color=C_CM)
    ax[0].set_xlabel("gap g$_0$ (µm)"); ax[0].set_ylabel("quality factor Q")
    ax[0].set_title("Derived squeeze-film Q vs. gap")
    ax[1].plot(beta, [sf.skvor_attenuation(b) for b in beta], color=C_FEM)
    ax[1].set_xlabel(r"perforation ratio $\beta=r_0/r_c$")
    ax[1].set_ylabel(r"Škvor $K(\beta)$")
    ax[1].set_title("Backplate perforation attenuation")
    fig.tight_layout(); _save(fig, "fig4_squeeze_film")


# --------------------------------------------------- fig 5: noise / SNR map
def fig_noise_snr():
    radii = np.linspace(0.6e-3, 1.4e-3, 40)
    gaps = np.array([4e-6, 6e-6, 8e-6])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for g in gaps:
        snr = [ref_model(radius=a, gap=g).snr() for a in radii]
        ax[0].plot(radii * 1e3, snr, label=f"g$_0$={g*1e6:.0f} µm")
    ax[0].set_xlabel("diaphragm radius (mm)"); ax[0].set_ylabel("SNR (dB(A))")
    ax[0].set_title("Predicted SNR vs. geometry"); ax[0].legend(fontsize=9)
    # anchor band from commercial datasheets
    try:
        anc = _load_anchors()
        lo, hi = min(anc["snr"]), max(anc["snr"])
        ax[0].axhspan(lo, hi, color="grey", alpha=0.15)
        ax[0].text(radii[0] * 1e3, hi, "commercial analog MEMS mics",
                   fontsize=8, color="grey", va="bottom")
    except Exception:
        pass
    # EIN spectrum (white, band-limited) for the reference design
    cm = ref_model()
    f = np.linspace(20, 20e3, 500)
    Sp = cm.input_noise_psd() * np.ones_like(f)
    ax[1].semilogx(f, 10 * np.log10(Sp) + 120, color=C_CM)  # dB re 1 uPa/rtHz
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel(r"input noise (dB re 1 µPa/$\sqrt{Hz}$)")
    ax[1].set_title(f"Thermal-mechanical noise floor (SNR={cm.snr():.0f} dB(A))")
    fig.tight_layout(); _save(fig, "fig5_noise_snr")


# --------------------------------------------------- fig 6: THD vs SPL
def fig_thd():
    cm = ref_model()
    ft = 1000.0
    fs = 200e3
    t = np.arange(0, 20e-3, 1 / fs)
    spls = np.array([80, 90, 94, 100, 110, 120, 124, 128])
    thd = []
    for spl in spls:
        p_amp = np.sqrt(2) * 20e-6 * 10 ** (spl / 20)
        v = cm.transient(t, p_amp * np.sin(2 * np.pi * ft * t))
        thd.append(_thd(v[t > 5e-3], fs, ft))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.semilogy(spls, np.array(thd) * 100, "o-", color=C_CM)
    ax.axhline(1.0, ls="--", color="grey"); ax.text(81, 1.1, "1% THD", fontsize=8)
    ax.axhline(10.0, ls="--", color="grey"); ax.text(81, 11, "10% THD", fontsize=8)
    ax.set_xlabel("sound pressure level (dB SPL)"); ax.set_ylabel("THD (%)")
    ax.set_title("Large-signal distortion (1 kHz, nonlinear transient)")
    fig.tight_layout(); _save(fig, "fig6_thd")


# --------------------------------------------------- fig 7: commercial anchors
def fig_anchors():
    anc = _load_anchors()
    # model design cloud: sweep radius, gap, back-volume
    S, N = [], []
    for a in np.linspace(0.6e-3, 1.4e-3, 6):
        for g in (4e-6, 6e-6, 8e-6):
            for vb in (5e-9, 10e-9, 30e-9):
                m = ref_model(radius=a, gap=g, cavity_volume=vb)
                m.bias = 0.6 * m.pull_in_voltage()   # consistent safe operating point
                S.append(20 * np.log10(m.sensitivity())); N.append(m.snr())
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.scatter(S, N, s=14, color=C_CM, alpha=0.5, label="compact-model designs")
    ax.scatter(anc["sens"], anc["snr"], s=80, marker="D", color=C_FEM,
               edgecolor="k", zorder=5, label="commercial analog MEMS mics")
    for s, n, p in zip(anc["sens"], anc["snr"], anc["part"]):
        ax.annotate(p.split()[-1], (s, n), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xlabel("sensitivity (dBV re 1 V/Pa)"); ax.set_ylabel("SNR (dB(A))")
    ax.set_title("Model design space vs. commercial parts")
    ax.annotate("model SNR = mechanical-thermal bound\n(commercial parts lower: read-out\nelectronics dominate their noise)",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7.5,
                color="dimgray", va="bottom")
    ax.legend(fontsize=9, loc="upper right"); fig.tight_layout()
    _save(fig, "fig7_anchors")


# --------------------------------------------------- helpers
def _load_anchors():
    part, sens, snr = [], [], []
    with open(os.path.join(DATA, "commercial_anchors.csv")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("part,"):
                continue
            c = line.split(",")
            part.append(c[0]); sens.append(float(c[2])); snr.append(float(c[3]))
    return dict(part=part, sens=sens, snr=snr)


def _thd(v, fs, f0):
    v = v - v.mean()
    w = np.hanning(len(v))
    V = np.abs(np.fft.rfft(v * w))
    fr = np.fft.rfftfreq(len(v), 1 / fs)
    def amp(fk):
        i = np.argmin(np.abs(fr - fk))
        return V[max(i - 2, 0):i + 3].max()
    fund = amp(f0)
    harm = np.sqrt(sum(amp(n * f0) ** 2 for n in range(2, 8)))
    return harm / fund if fund > 0 else 0.0


def _save(fig, name):
    path = os.path.join(FIG, name + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


if __name__ == "__main__":
    fig_validation()
    fig_frequency_response()
    fig_speed_accuracy()
    fig_squeeze_film()
    fig_noise_snr()
    fig_thd()
    fig_anchors()
