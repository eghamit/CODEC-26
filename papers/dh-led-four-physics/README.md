# DH-LED four-physics paper (CODEC, Track 2: Optoelectronic and Photonic Devices)

A unified 2-D FEM drift-diffusion simulation of a GaAs/AlGaAs double-heterostructure
LED coupling **four** physical effects — heterojunction band offsets, self-consistent
Schrödinger–Poisson subband confinement, Fermi–Dirac degeneracy, and radiative
recombination — with a controlled per-physics ablation quantifying each one.

## Contents

- `paper.tex` — the paper in **IEEE conference format** (`IEEEtran`, `[conference]`).
  Compile with `pdflatex paper && pdflatex paper` (run twice for cross-references);
  needs `figures/*.png`. The compiled `paper.pdf` (5 pages) is included.
- `paper.pdf` — compiled output of `paper.tex`.
- `paper.md` — the same content in Markdown (readable diff / web view). The
  literature review (§II) sets out the research gap and the advantage of the
  proposed work (Table I).
- `figures/` — the five figures, regenerated from the solver:
  - `fig1_iv.png` — total + radiative I–V (full model)
  - `fig2_iqe_ablation.png` — IQE vs. current, four-config ablation (the DH advantage)
  - `fig3_iv_ablation.png` — I–V ablation (turn-on shift)
  - `fig4_confinement.png` — equilibrium electron density + potential (homo vs. hetero vs. SP)
  - `fig5_subbands.png` — the four confined electron subbands
- `data/ablation.json` — raw ablation results (V_bi, I–V, IQE, λ, subbands per config).
- `code/` — reproduction scripts (`ablation.py`, `make_figs.py`).

## Headline results (300 nm × 10 nm DH-LED, 300 K)

| Configuration | V_bi [V] | Peak IQE | λ [nm] |
|---|:--:|:--:|:--:|
| Homojunction (Boltzmann, classical) | 1.389 | 0.901 | 873.1 |
| + Heterojunction | 1.750 | 0.908 | 873.1 |
| + Fermi–Dirac | 1.765 | 0.897 | 873.1 |
| + Schrödinger–Poisson (full) | 1.766 | 0.897 | 873.1 |

Electron subbands (full model): E = [−1.031, −0.971, −0.789, −0.478] eV.

## Reproducing

The scripts require the open **DDM.SPC** solver (`pip install -e ".[all]"` in the
[eghamit/Schrodinger](https://github.com/eghamit/Schrodinger) repository). Then, from
this `code/` directory:

```bash
python ablation.py     # runs the four configurations -> ablation.json (~2-3 min)
python make_figs.py    # reads ablation.json -> figures/
```

> The physics engine lives in the separate Schrödinger repository; this directory
> holds only the paper, its figures, data, and the scripts that drive the solver.
