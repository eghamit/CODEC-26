"""Transducer + CMOS read-out co-simulation in ngspice.

The compact model is emitted as a small-signal SPICE subcircuit (the linear
electro-mechanical 2-port: a force-driven mechanical R-L-C coupled to the
electrical port through the transduction factor Gamma and the bias capacitance
C0).  ngspice then simulates it together with a real CMOS read-out -- something
a full FEM transducer model cannot do inside a circuit simulator.

Three experiments:
  * ``verify_open_circuit`` -- open electrical port; the SPICE magnitude response
    must match the Python ``transducer_tf_midband`` (this is the golden check
    that ngspice implements the compact model correctly).
  * ``source_follower`` -- classic analog MEMS-mic front-end: NMOS source
    follower with a giga-ohm bias resistor; shows the electrical high-pass and
    buffer loading the full FEM model cannot capture.
  * ``charge_amplifier`` -- op-amp charge integrator (virtual-ground read-out).

Requires the ``ngspice`` binary on PATH.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

NGSPICE = "ngspice"


# ----------------------------------------------------------------- runner
def _run(netlist):
    """Run an ngspice batch netlist; return dict of columns from wrdata."""
    with tempfile.TemporaryDirectory() as d:
        cir = os.path.join(d, "run.cir")
        out = os.path.join(d, "out.data")
        netlist = netlist.replace("__OUT__", out)
        with open(cir, "w") as f:
            f.write(netlist)
        res = subprocess.run([NGSPICE, "-b", cir], capture_output=True,
                             text=True, timeout=120)
        if not os.path.exists(out):
            raise RuntimeError("ngspice produced no output:\n" + res.stderr)
        data = np.loadtxt(out)
    return data


# ------------------------------------------------------------- subcircuit
def linear_subckt(params, name="mems_lin"):
    """Small-signal MEMS transducer as a SPICE subcircuit (pin -> out).

    Mechanical loop current == diaphragm velocity; a CCCS injects Gamma*velocity
    into the electrical node loaded by C0.  (Electrical back-action is a
    higher-order term retained in the full Verilog-A model; omitted here so the
    open-circuit response equals the analytic transducer TF for verification.)
    """
    p = params
    return f"""* {name}: linear electro-mechanical transducer
.subckt {name} pin out
Bf   force 0  V = {p['Aeff']:.6e}*V(pin)
Lm   force nm {p['M']:.6e}
Rm   nm   nv {p['Cdamp']:.6e}
Vs   nv   nq 0
Cm   nq   0  {1.0/p['Knet']:.6e}
Fmot 0 out Vs {p['Gamma']:.6e}
C0   out  0  {p['C0']:.6e}
.ends {name}
"""


# ----------------------------------------------------------- experiments
def verify_open_circuit(cm, f_lo=10.0, f_hi=2e5, ndec=30):
    """Return (freqs, |Vout/Pin|_spice, |Vout/Pin|_python)."""
    p = cm.spice_parameters()
    net = linear_subckt(p) + f"""* open-circuit transducer AC
X1 pin out mems_lin
Rhuge out 0 1e15
Vpin pin 0 AC 1
.ac dec {ndec} {f_lo:g} {f_hi:g}
.control
run
wrdata __OUT__ v(out)
.endc
.end
"""
    d = _run(net)
    freqs = d[:, 0]
    mag = np.hypot(d[:, 1], d[:, 2])
    py = np.abs(cm.transducer_tf_midband(freqs))
    return freqs, mag, py


def source_follower(cm, f_lo=1.0, f_hi=2e5, ndec=20, rbias=10e9, vdd=3.3):
    """NMOS source-follower read-out; return (freqs, |Vout/Pin|)."""
    p = cm.spice_parameters()
    vg = 0.9                                    # gate DC operating point
    net = linear_subckt(p) + f"""* MEMS + NMOS source-follower read-out
X1 pin g mems_lin
Vpin pin 0 AC 1
* bias network on the high-impedance sense node
Rbias g vbias {rbias:.3e}
Vbias vbias 0 {vg}
* NMOS source follower
.model nsf nmos (level=1 vto=0.45 kp=250u lambda=0.02 cgso=1e-11 cgdo=1e-11)
M1 vdd g out 0 nsf w=40u l=0.5u
Vdd vdd 0 {vdd}
Ibias out 0 20u
Rload out 0 100k
.ac dec {ndec} {f_lo:g} {f_hi:g}
.control
run
wrdata __OUT__ v(out)
.endc
.end
"""
    d = _run(net)
    return d[:, 0], np.hypot(d[:, 1], d[:, 2])


def charge_amplifier(cm, f_lo=1.0, f_hi=2e5, ndec=20, cf=0.5e-12, rf=1e12):
    """Op-amp charge-integrator read-out; return (freqs, |Vout/Pin|)."""
    p = cm.spice_parameters()
    net = linear_subckt(p) + f"""* MEMS + op-amp charge amplifier
X1 pin inv mems_lin
Vpin pin 0 AC 1
* ideal op-amp: non-inverting = gnd, inverting = inv, out = vo
Eamp vo 0 0 inv 1e6
Cf inv vo {cf:.3e}
Rf inv vo {rf:.3e}
Rout vo 0 100k
.ac dec {ndec} {f_lo:g} {f_hi:g}
.control
run
wrdata __OUT__ v(vo)
.endc
.end
"""
    d = _run(net)
    return d[:, 0], np.hypot(d[:, 1], d[:, 2])


if __name__ == "__main__":
    from compact_model import CompactMicModel
    cm = CompactMicModel.circular(1.0e-3, 6.0, 3100 * 0.5e-6, 6e-6, bias=10.0,
                                  cavity_volume=10e-9, vent_radius=3e-6,
                                  vent_length=30e-6)
    f, spice, py = verify_open_circuit(cm)
    # compare in the mid-band (below resonance)
    mask = f < 0.5 * cm.resonance
    err = np.max(np.abs(spice[mask] - py[mask]) / py[mask])
    print(f"open-circuit ngspice vs Python  max mid-band rel error = {err:.2e}")
    print(f"  mid-band |H| ngspice = {spice[mask][0]*1e3:.3f} mV/Pa, "
          f"python = {py[mask][0]*1e3:.3f} mV/Pa")
