# JMST (Springer Nature) version

This folder holds the paper in the **Springer Nature journal template**
(`sn-jnl.cls`, v3.1 Dec 2024) for submission to the **Journal of Marine Science
and Technology (JMST)**. The IEEE conference version lives one level up
(`../paper.tex`).

Reference style: **Math and Physical Sciences, numbered** (`sn-mathphys-num`).

## Files

```
jmst-springer/
├── manuscript.tex        # the paper in Springer format
├── manuscript.pdf        # compiled output (13 pages, single column)
├── references.bib        # BibTeX bibliography (22 entries with DOIs)
├── sn-jnl.cls            # Springer Nature class
├── sn-mathphys-num.bst   # BibTeX style used here
└── figures/              # the seven figures (shared with the IEEE version)
```

The figures and all numbers are produced by the reproducible pipeline in
`../code/` — see the top-level `../README.md`.

## Compiling

Requires a LaTeX distribution with the Springer class dependencies (e.g. TeX
Live `texlive-latex-base texlive-latex-recommended texlive-latex-extra
texlive-science`, which provide `cuted`, `stfloats`, `siunitx`, `booktabs`,
etc.). Then:

```bash
pdflatex manuscript
bibtex   manuscript
pdflatex manuscript
pdflatex manuscript
```

## Notes for submission

- Fill in real author names, affiliations, ORCID and corresponding-author
  e-mail in the `\author` / `\affil` block.
- The study is now built on **real towing-tank data** (the Delft Systematic
  Yacht Hull Series, 308 experiments), benchmarked against the traditional
  polynomial regression, with calibrated Gaussian-process uncertainty — the
  substantive upgrade needed for a marine-engineering journal.
- Remaining strengthening options a reviewer may still ask for: validation on
  **full-form commercial-hull** data (the DSYHS is a yacht series), a
  multi-fidelity extension fusing CFD with the experimental labels, and a
  confirmatory CFD/tank check of a surrogate-selected optimum. These are
  described as future work in the manuscript's roadmap and limitations sections.
