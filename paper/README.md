# IEEE-style paper

An 11-page IEEE journal-style manuscript (IEEEtran) on the open, validated
FEM drift-diffusion + Schrodinger-Poisson framework for ISFET/BioFET biosensor
simulation with a self-consistent electrolyte double-layer model.

Contents: extensive literature review (42 references), device + electrolyte
theory with equations and two algorithms, a solver validation against analytic
physics, ISFET/BioFET results (sub-Nernstian pH response, Debye-screening limit),
a biosensor design map, applications to disease detection, and two appendices.

## Files
- `main.tex`  - manuscript source (IEEEtran, two-column)
- `refs.bib`  - bibliography (45 entries, 42 cited)
- `main.pdf`  - compiled manuscript (11 pages)
- `figures/`  - all figures (a TikZ schematic is inline in main.tex)

## Build
```bash
make          # pdflatex + bibtex + pdflatex x2
```
Requires: texlive-latex-base, -recommended, -extra, -fonts-recommended,
texlive-publishers (IEEEtran), texlive-science (siunitx).

## Note
The author block is a placeholder and must be completed before submission.
Oxide site-binding parameters are representative/illustrative values; the device
solver is validated against analytic physics but not yet against measured data.
