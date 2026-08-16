# Data source: Delft Systematic Yacht Hull Series (DSYHS)

`yacht_hydrodynamics.data` is the **UCI Machine Learning Repository "Yacht
Hydrodynamics" data set** — 308 full-scale towing-tank experiments from the
Delft Ship Hydromechanics Laboratory (the Delft Systematic Yacht Hull Series).
These are **real physical measurements**, not synthetic values.

- Original source: UCI ML Repository, dataset 243
  (https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics).
- Retrieved here from a public GitHub mirror
  (`raw.githubusercontent.com/Laxman-Kumar/Yacht-Hydrodynamics`) because the UCI
  host is not reachable from the build environment; the file was byte-verified
  against a second independent mirror
  (`danielmserna/DL-Yacht-Hydrodynamics-Python`) — the two agree exactly
  (308 rows, order-independent).
- Primary reference: J. Gerritsma, R. Onnink, A. Versluis, "Geometry, resistance
  and stability of the Delft Systematic Yacht Hull Series," *International
  Shipbuilding Progress*, vol. 28, no. 328, 1981.

## Columns (whitespace-delimited, no header)

| # | Symbol | Variable | Units |
|---|--------|----------|-------|
| 1 | LCB | Longitudinal position of the centre of buoyancy | adim. |
| 2 | Cp  | Prismatic coefficient | adim. |
| 3 | L/∇^(1/3) | Length–displacement ratio | adim. |
| 4 | B/T | Beam–draught ratio | adim. |
| 5 | L/B | Length–beam ratio | adim. |
| 6 | Fn  | Froude number | adim. |
| 7 | **Rr** | **Residuary resistance per unit weight of displacement (TARGET)** | adim. |

The target is the residuary (chiefly wave-making) resistance; frictional
resistance is handled separately by classical correlation lines and is not part
of this dataset. Froude number spans 0.125–0.45 in steps of 0.025.
