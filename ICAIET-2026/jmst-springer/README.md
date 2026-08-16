# JMST (Springer Nature) version

This folder holds the **same paper reformatted in the Springer Nature journal
template** (`sn-jnl.cls`, v3.1 Dec 2024) for submission to the
**Journal of Marine Science and Technology (JMST)**. The IEEE conference version
lives one level up (`../paper.tex`).

Reference style: **Math and Physical Sciences, numbered** (`sn-mathphys-num`).

## Files

```
jmst-springer/
├── manuscript.tex        # the paper in Springer format
├── manuscript.pdf        # compiled output (15 pages, single column)
├── references.bib        # BibTeX bibliography (20 entries with DOIs)
├── sn-jnl.cls            # Springer Nature class (from the template package)
├── sn-mathphys-num.bst   # BibTeX style used here
└── figures/              # the six figures (shared with the IEEE version)
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
- The `\documentclass` line currently uses the default `sn-mathphys-num` style;
  switch to another bundled style only if JMST's Instructions for Authors ask
  for it.
- JMST is a naval-architecture / marine-engineering journal. The most important
  strengthening step before submission is to **replace the semi-empirical
  ground-truth oracle with real CFD or published experimental (e.g. DSYHS,
  KCS/KVLCC2) resistance data** and add a **Holtrop–Mennen baseline**. The
  current draft is transparent about this being synthetic-label data; that
  limitation is the one a marine-domain reviewer is most likely to press on.
