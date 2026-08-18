# BioFET double-layer front-end (G-DL)

Electrolyte **double-layer + site-binding** gate model that turns an ISFET/BioFET
from an *ideal* MOS threshold sensor into a *realistic* one. It supplies the
physics the bare device solver lacks — the surface chemistry and the electrical
double layer — so the analyte-to-gate coupling becomes **sub-unity**, which is
what real electrolyte-gated biosensors show.

This is the implementation of extension **G-DL** described in
`../solver-applications-exploration/applications_exploration.md`.

## Why it matters

The device-only ISFET examples (in `../solver-applications-exploration/`) give an
analyte-to-threshold coupling of **exactly 1.000** — the ideal limit, because the
solver has no electrolyte. Two effects break that idealisation in a real device,
and both are captured here:

1. **Sub-Nernstian pH response.** Site-binding of protons at the functionalised
   oxide, screened by the double layer, gives `dψ₀/dpH < 59.5 mV/pH` (the
   Nernst limit). The reduction factor `α = sensitivity / 59.5 mV` depends on the
   surface buffer capacity.
2. **Debye-screening detection limit.** A charged biomolecule bound a distance `d`
   from the surface is screened by `exp(-d/λ_D)`. As ionic strength rises, `λ_D`
   shrinks below `d` and the response collapses — the central limitation of
   BioFET biosensing in physiological media.

## The model

- **Site-binding (2-pK amphoteric)** surface charge `σ₀(ψ₀)`, with the surface
  proton activity Boltzmann-shifted by `ψ₀` (Yates–Levine–Healy /
  Fung–Ko / van Hal–Bergveld 1996).
- **Gouy–Chapman–Stern** double layer: a Stern (Helmholtz) capacitance in series
  with the diffuse layer `σ_d(ψ_d) = -√(8εkT n₀) sinh(qψ_d/2kT)`.
- **Charge balance** `σ₀ = -σ_d` with `ψ₀ - ψ_d = σ₀/C_stern`, solved
  self-consistently for `ψ₀` at each pH.
- **Debye–Hückel screening** of a bound-biomolecule charge sheet.

The surface potential `ψ₀` then enters the device gate as an effective
work-function shift `ΔΦ = -ψ₀`, driving the same DDM.SPC `I_D(V_G)` transfer-curve
solve used in the sibling examples.

## Files

| file | what it is |
|---|---|
| `double_layer.py` | the G-DL physics module (pure numpy/scipy, self-contained) |
| `biosensor_isfet_gdl.py` | end-to-end example: pH response + Debye screening, with an optional full DDM.SPC device solve |

## Representative results

pH sensitivity at pH 7, I = 0.15 M (physiological); Nernst limit = 59.2 mV/pH:

| oxide | mV/pH | α (coupling) |
|---|--:|--:|
| Ta₂O₅ | 57.4 | 0.97 |
| Al₂O₃ | 49.5 | 0.84 |
| Si₃N₄ | 46.9 | 0.79 |
| SiO₂  | 31.6 | 0.53 |

BioFET Debye screening (bound charge −0.01 C/m² at d = 3 nm): response falls from
−101 mV at 1 mM to −0.25 mV at physiological ionic strength (~400× as λ_D drops
below d).

> The per-oxide `(N_s, pKa, pKb)` triples in `double_layer.py` are *illustrative*
> values chosen to reproduce the known ISFET ordering and sensitivity ranges; they
> are meant to be re-fit to a given fabrication, not taken as material constants.

## Run

```bash
pip install numpy scipy                 # module only
python biosensor_isfet_gdl.py           # electrolyte analysis (fast)
python biosensor_isfet_gdl.py --plot    # + figures
python biosensor_isfet_gdl.py --device  # + full DDM.SPC transfer-curve chain (slow; needs DDM.SPC + gmsh + the mosfet_v3 mesh)
```

The `--device` run imports `DDM_SPC` and reads `meshing/mosfet_v3.msh`; run it from
a checkout of the Schrödinger/DDM.SPC solver (or point the mesh path at one).
