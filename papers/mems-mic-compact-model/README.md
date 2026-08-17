# Capacitive MEMS microphone: closed-form compact model + Verilog-A/SPICE co-simulation

A physically-derived, largely **closed-form** compact model of a capacitive MEMS
microphone and its **Verilog-A / SPICE** realization, for transducer + CMOS
read-out **co-simulation** — the microelectronics-modeling framing of the
transducer, targeting *Microelectronics Journal* / CODEC "device modeling and
device physics."

The transducer's tensioned-membrane electromechanical PDE is reduced by a
single-mode Galerkin projection to a lumped electro-mechano-acoustic model. For
the circular diaphragm the modal parameters **and** the gap capacitance are
closed-form:

- fundamental shape is exactly parabolic → `m = ρₛA/3`, `k = 2πT`, `A_eff = A/2`
- **gap capacitance** `C(q) = (εA/q)·ln(g₀/(g₀−q))` — one elementary, analytically
  differentiable expression → analytic sensitivity and pull-in
- **squeeze-film damping** from the Škvor annular-cell Reynolds solution → the
  quality factor is *derived*, not fitted
- **back-cavity + vent** acoustic network → low-frequency high-pass corner
- **fluctuation–dissipation** noise → equivalent input noise and SNR

## Contents

- `paper.tex` — IEEE conference paper (`IEEEtran`); compile `pdflatex paper && pdflatex paper` (needs `figures/*.png`).
- `paper.md` — the same content in Markdown (readable / web view).
- `code/`
  - `fem_reference.py` — full 2-D FE electromechanical membrane (the ground truth).
  - `compact_model.py` — the reduced-order compact model (mirrors the Verilog-A).
  - `squeeze_film.py` — derived Škvor squeeze-film damping.
  - `mems_mic.va` — **Verilog-A** macromodel (for Spectre/Xyce/ADS).
  - `cosim_ngspice.py` — emits SPICE subcircuit + CMOS read-out, runs ngspice.
  - `verification.py` — FEM-vs-compact accuracy + speed study → `data/results.json`.
  - `make_figures.py` — regenerates `figures/*.png`.
  - `test_models.py` — pytest validation suite (`pytest.ini` registers the `slow` mark).
- `data/` — `results.json`, `commercial_anchors.csv`, `DATA_SOURCE.md`.
- `figures/` — the seven figures.

## Headline results (1 mm-radius Si₃N₄ diaphragm, 6 µm gap, 10 V bias, 10 mm³ cavity)

| Quantity | Value |
|---|--:|
| Rest capacitance C₀ | 4.64 pF |
| Modal mass / stiffness | 1.62 nkg / 37.7 N·m⁻¹ |
| Derived quality factor Q | 6.2 |
| Pull-in voltage | 15.6 V |
| Resonance / high-pass corner | 32.4 kHz / 0.13 Hz |
| Sensitivity | 21.2 mV/Pa (−33.5 dBV/Pa) |
| Mechanical-thermal SNR | 83 dB(A) |

### Compact model vs. full FEM

| Design | Sensitivity err | f₀ err | Pull-in err | Speed-up |
|---|--:|--:|--:|--:|
| D1 (0.8 mm, T4) | 0.74 % | 1.5 % | 2.0 % | 3.6×10⁵ |
| D2 (1.0 mm, T6) | 0.77 % | 1.5 % | 2.0 % | 5.5×10⁵ |
| D3 (1.2 mm, T8) | 0.78 % | 1.5 % | 2.0 % | 4.8×10⁵ |

The SPICE macromodel reproduces the analytic transducer to **2×10⁻⁸** in the
mid-band; ngspice then co-simulates the transducer with an NMOS source-follower
and an op-amp charge-amplifier front-end.

## Reproducing

Requires `numpy`, `scipy`, `matplotlib`, `pytest`, and the `ngspice` binary
(for the co-simulation). From `code/`:

```bash
python -m pytest -q            # validation suite (full FEM cross-checks + ngspice)
python verification.py         # -> ../data/results.json  (full FEM, ~3 min)
python make_figures.py         # -> ../figures/*.png
python cosim_ngspice.py        # open-circuit ngspice-vs-Python check
```

The physics is self-contained in `code/` (no external solver dependency); the
Verilog-A device `mems_mic.va` targets any Verilog-AMS simulator.

## Scope and honesty notes

- The full-FEM comparison is **bare-diaphragm** (no cavity), so it isolates the
  modal-reduction error; the cavity/vent/noise are lumped physics the compact
  model adds and the FE membrane solver does not provide.
- The reported SNR is the **mechanical-thermal bound**; real parts are lower
  because read-out electronics dominate their noise — which is precisely why the
  co-simulation matters. `data/commercial_anchors.csv` holds published *headline*
  datasheet specs (to be re-verified per revision before camera-ready); no
  measured data is claimed or fabricated.
- Author/affiliation and the target venue in `paper.tex` are placeholders.
