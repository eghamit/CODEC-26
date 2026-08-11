# A Unified Four-Physics Finite-Element Simulation of a GaAs/AlGaAs Double-Heterostructure LED: Coupled Heterojunction Electrostatics, Self-Consistent Schrödinger–Poisson Confinement, Fermi–Dirac Degeneracy and Radiative Recombination

*Prepared for submission to the 9th International Conference on Computers and Devices for Communication (CODEC).
Track: Optoelectronic and Photonic Devices.*

---

## Abstract

Accurate device-level modelling of a modern light-emitting diode (LED) requires
four physical effects that classical drift-diffusion (DD) omits or treats
crudely: (i) the heterojunction band-edge offsets that confine injected carriers
in the active region, (ii) the quantum-mechanical subband confinement that
reshapes the carrier distribution in a thin active layer, (iii) Fermi–Dirac
degeneracy in the heavily-doped cladding, and (iv) the radiative recombination
that produces the light. Commercial technology-CAD (TCAD) tools can combine
these, but they are proprietary and non-reproducible; the available open-source
device simulators typically address only a subset, most commonly through the
density-gradient quantum approximation rather than a genuine self-consistent
Schrödinger–Poisson (SP) subband solve, and rarely for a heterostructure
optoelectronic emitter. **This paper presents, in a single open, test-validated
2-D finite-element (FEM) solver, the fully-coupled combination of all four
effects for a GaAs/AlGaAs double-heterostructure (DH) LED, and quantifies each
effect through a controlled per-physics ablation.** The solver is written in the
quasi-Fermi-potential formulation and solved by a fully-coupled damped-Newton
method with an exact analytic Jacobian; the SP correction and the Fermi–Dirac
degeneracy shift are refreshed by a common outer self-consistency (Gummel) loop
that preserves the inner Newton's quadratic convergence. On a 300 nm × 10 nm DH
device the model reproduces the expected physics quantitatively: the
heterojunction band offsets raise the built-in potential from 1.389 V
(homojunction) to 1.750 V and shift the current turn-on so that the DH device
reaches a given internal quantum efficiency (IQE) at roughly two orders of
magnitude lower drive current than the homojunction; the self-consistent SP loop
resolves four confined electron subbands in the active layer and sets the carrier
peak back from the heterointerface; the Fermi–Dirac correction removes the
Boltzmann over-count of the degenerate cladding; and the radiative channel yields
a peak IQE of 0.90 at an emission wavelength of 873 nm consistent with the GaAs
active-region band gap. All results are reproducible from the accompanying open
scripts and a unit-test suite that checks the solver against analytic
semiconductor physics.

**Keywords:** device simulation, drift-diffusion, Schrödinger–Poisson,
heterojunction, Fermi–Dirac statistics, radiative recombination, light-emitting
diode, finite-element method, open-source TCAD, internal quantum efficiency.

---

## I. Introduction

The double heterostructure is the foundational concept of modern optoelectronics:
by cladding a narrow-gap active region between two wider-gap layers, injected
electrons and holes are confined together, dramatically raising the radiative
efficiency of LEDs and enabling room-temperature laser operation — the
contribution recognised by the 2000 Nobel Prize in Physics [1]. Predictive
simulation of such a device is inherently a *multi-physics* problem. The
electrostatics are governed by position-dependent permittivity and band-edge
offsets; the thin active region confines carriers quantum-mechanically into
subbands; the degenerately-doped cladding violates Boltzmann statistics; and the
useful output — light — is set by the radiative recombination rate in the active
region. A simulator that captures only the classical drift-diffusion transport
will mispredict the turn-on, the carrier distribution, and the efficiency.

Technology-CAD (TCAD) frameworks such as Synopsys Sentaurus Device and Silvaco
Atlas/Victory can, in principle, combine all of these models. They are, however,
closed-source and licence-restricted: their model formulations and default
parameterisations cannot be inspected, their results cannot be reproduced by a
reader without the same commercial licence, and their internal coupling
strategies are not open to scrutiny. For research communication — where
reproducibility is increasingly a requirement — and for teaching, an open,
inspectable alternative is valuable.

This paper contributes such an alternative for the specific and demanding case of
a DH-LED. We use **DDM.SPC**, an open 2-D FEM drift-diffusion solver, and show
that a *single* self-consistent solve can couple **four** physical effects that
are usually studied in isolation:

1. **Heterojunction electrostatics** — Anderson's electron-affinity rule for the
   conduction- and valence-band offsets, position-dependent permittivity and
   mobility;
2. **Self-consistent Schrödinger–Poisson confinement** — a genuine effective-mass
   subband eigenproblem solved per confinement slice, not a density-gradient
   surrogate;
3. **Fermi–Dirac statistics** — a degeneracy correction for the heavily-doped
   cladding;
4. **Radiative (bimolecular) recombination** — with post-processed optical
   output (IQE, emission wavelength, optical power).

Crucially, we do not merely run the combined model; we **isolate the contribution
of each effect** through a controlled ablation on the *same* device and mesh, so
that the shift each physics ingredient produces in the built-in potential, the
turn-on, the carrier profile and the efficiency can be read off directly. To our
knowledge this is the first open, reproducible demonstration of the four coupled
in one FEM solve for a heterostructure emitter.

## II. Related Work and Research Gap

We organise prior work into three groups and identify, for each, what is missing
relative to the present contribution.

**A. Commercial TCAD.** Synopsys Sentaurus Device and Silvaco Atlas/Victory are
the industry-standard drift-diffusion device simulators and support
heterojunctions, quantum corrections, Fermi–Dirac statistics and optical
recombination. They are, however, **proprietary and non-reproducible**: neither
the model source nor the solver internals are open, and reproducing a published
result requires an identical commercial licence and version. Quantum confinement
in these tools is most commonly applied through the *density-gradient*
approximation rather than a direct Schrödinger solve.

**B. Open-source device simulators.** A small number of open device simulators
exist. **DEVSIM** [2] is a finite-volume drift-diffusion simulator designed to
compare with commercial TCAD; it offers the density-gradient method for quantum
effects near insulator boundaries, but not a self-consistent Schrödinger–Poisson
subband solve, and its emphasis is silicon electronics rather than heterostructure
optoelectronics. **Genius-TCAD (open edition)** [3] provides 2-D FEM
drift-diffusion with lattice heating and a range of mobility models, but no
self-consistent SP subband model. **Charon** [4] (Sandia) is a large C++/Trilinos
FEM device code aimed principally at radiation and silicon-technology modelling.
**SEMIDV** [5] is a recent Python Poisson-drift-diffusion simulator that
introduces quantum corrections, again primarily for compact CMOS modelling. None
of these is presented as a coupled heterojunction + SP + Fermi–Dirac + radiative
solve for a DH-LED with a per-physics ablation.

**C. Coupled Schrödinger–Poisson / quantum drift-diffusion methods.**
Self-consistent Schrödinger–Poisson coupling with drift-diffusion transport
("quantum drift-diffusion") is well established as a methodology [6], and is used
in nanowire-MOSFET and quantum-well studies, and in dedicated SP solvers for
confined electron gases. This literature concentrates on the *transport/electronic*
problem (MOSFET channels, 2-DEGs, single-photon sources); it is generally not
integrated with a radiative-recombination optical read-out for an LED, nor
delivered as an open, reproducible package with the heterojunction and
degeneracy physics coupled in the *same* solve.

**Research gap.** Table I summarises the position. Each prior category provides a
subset of the required physics or lacks openness/reproducibility. The specific
gap addressed here is a **single open, test-validated FEM solve that couples
heterojunction electrostatics, a genuine self-consistent Schrödinger–Poisson
subband correction, Fermi–Dirac degeneracy, and radiative recombination for a
double-heterostructure LED, with a controlled ablation that quantifies each
effect's contribution to the built-in potential, turn-on and IQE.**

**Table I. Capability comparison (● present, ◐ partial/approximate, ○ absent).**

| Capability | Sentaurus / Silvaco | DEVSIM [2] | Genius-open [3] | SEMIDV [5] | SP/QDD studies [6] | **This work** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Open-source / reproducible | ○ | ● | ● | ● | ◐ | ● |
| 2-D FEM drift-diffusion | ● | ◐ (FV) | ● | ◐ | ◐ | ● |
| Heterojunction band offsets | ● | ◐ | ◐ | ○ | ◐ | ● |
| Quantum confinement | ◐ (dens.-grad.) | ◐ (dens.-grad.) | ○ | ◐ | ● (SP) | ● (self-consistent SP) |
| Fermi–Dirac degeneracy | ● | ● | ● | ◐ | ◐ | ● |
| Radiative recomb. + optical read-out | ● | ○ | ◐ | ○ | ○ | ● |
| **All four coupled + per-physics ablation for a DH-LED** | ◐ (closed) | ○ | ○ | ○ | ○ | **●** |

**Advantage of the proposed work.** Relative to the closed commercial tools it is
fully open and reproducible (every number in this paper regenerates from the
accompanying scripts and passes an analytic-physics unit-test suite); relative to
the open tools it adds a genuine self-consistent Schrödinger–Poisson subband
correction (not the density-gradient surrogate) and couples it with
heterojunction, degeneracy and radiative physics in one solve; and relative to
the SP/quantum-drift-diffusion literature it closes the loop to an optoelectronic
observable (IQE and emission wavelength) and, uniquely, provides a controlled
ablation that attributes the built-in-potential, turn-on and efficiency changes to
each individual physical mechanism.

## III. Device Structure and Physical Models

**Device.** The test device (Fig. 1 inset geometry) is a lateral DH-LED, 300 nm
along the transport direction *x* and 10 nm thick along the confinement direction
*y*. A GaAs active region occupies the middle third (100–200 nm) and is clad on
both sides by Al₀.₃Ga₀.₇As. The cladding is degenerately doped
(|N| ≈ 1×10²⁴ m⁻³ ≈ 10¹⁸ cm⁻³, p on the anode side, n on the cathode side); the
GaAs active region is lightly doped (≈ 10¹⁶ cm⁻³). A structured triangular mesh
(1037 nodes, 1920 elements) is used so that the confinement slices align with the
thin dimension.

**Primary unknowns.** The solver uses the normalised quasi-Fermi-potential
variables (De Mari intrinsic scaling [7]): the scaled electrostatic potential
*u* = ψ/V_T and the electron/hole quasi-Fermi potentials *v*, *w*. Carrier
densities follow algebraically, n = n_i e^{u−v}, p = n_i e^{w−u}, so they never
enter the linear algebra directly.

**(1) Heterojunction electrostatics.** With more than one material, a reference
material (the GaAs active region) anchors the scaling and every other region
enters through position-dependent permittivity ε(x)/ε_ref on the Poisson
operator, per-element mobilities µ(x)/µ_ref, and band-edge offsets from
Anderson's electron-affinity rule, which appear as fixed additive shifts in the
density exponents:

    n = n_i,ref · exp(u − v + band_n(x)),   p = n_i,ref · exp(w − u + band_p(x)),
    band_n = ln(N_c/N_c,ref) + (χ − χ_ref)/V_T,
    band_p = ln(N_v/N_v,ref) + [(χ_ref − χ) + (E_g,ref − E_g)]/V_T.

A single-material device gives identity coefficients and is byte-identical to the
classical homojunction solve.

**(2) Self-consistent Schrödinger–Poisson confinement.** Confinement is
one-dimensional (across the 10 nm body). The mesh is cut into slices along *y*
and, on each slice, the single-band effective-mass problem

    [ −(ħ²/2m*) d²/dy² + U(y) ] ψ_i = E_i ψ_i,   U_n = −qψ,  U_p = +qψ,

is solved by linear finite elements, giving the tridiagonal generalised
eigenproblem Hψ = E Mψ. The lowest few M-normalised subbands are summed with a
thermal (Boltzmann) occupation into a quantum density shape ρ(y) = Σ_i g_i|ψ_i|²,
g_i = e^{−(E_i−E_0)/k_BT}, which is renormalised so each slice carries exactly the
same charge as the classical density it replaces. The correction therefore only
*redistributes* carriers across the confinement direction (charge conserved,
Poisson well-posed) and enters the DD assembly as an additive quantum potential
Λ = ln(n_QM/n_cl) in the density exponent. Because Λ is frozen inside a Newton
step, the exact analytic Jacobian is unchanged. This is a self-consistent
predictor–corrector scheme with quasi-equilibrium subbands — the standard
device-TCAD way to add confinement — not a full non-equilibrium quantum-transport
(NEGF) model.

**(3) Fermi–Dirac statistics.** For the degenerate cladding the Fermi–Dirac
density n = N_c F_{1/2}(η) enters as a degeneracy shift ln F_{1/2}(η) − η of the
density exponent, refreshed by the same outer loop as the SP correction; it
vanishes in the non-degenerate limit (Boltzmann recovered exactly). F_{1/2} uses
the Bednarczyk analytic approximation [8] (< 0.4 % error).

**(4) Radiative recombination and optical output.** The light-emitting channel is
the bimolecular term R_rad = B(np − n_i²), added to the two continuity equations
with the (np − n_i²) prefactor (so it vanishes at equilibrium). Its exact
derivatives drop into the analytic Jacobian. After a converged forward-bias
solve, integrating the radiative rate over the device (using the *same*
quantum-corrected densities and quadrature) yields the spontaneous emission rate,
the radiative current I_rad = qR_rad, the internal quantum efficiency
IQE = R_rad/(R_rad + R_SRH + R_Auger), the emission wavelength from the
active-region band gap, and the optical power. Shockley–Read–Hall and Auger
recombination are enabled alongside as the non-radiative channels.

## IV. Numerical Method

The three coupled Galerkin weak forms (Poisson + two continuity equations) are
assembled into a global 3N×3N system and solved by a **fully-coupled damped-Newton
method** with the exact analytic 9×9 element Jacobian; all element integrals use
3-point Gauss quadrature. A robust thermal-equilibrium nonlinear-Poisson solve
provides the initial guess; each bias point warm-starts from the previous
solution; and the terminal current is extracted as the sum of the continuity
residuals over a contact's nodes (a conservative "test-function = 1" current
functional).

When the Schrödinger–Poisson correction and/or Fermi–Dirac statistics are active,
the inner Newton solve is wrapped in an **outer self-consistency (Gummel) loop**:
solve the frozen drift-diffusion system, recompute the confinement subbands (and
the degeneracy shift) from the updated potential, under-relax the correction, and
repeat until the maximum change in the quantum potential falls below tolerance.
Because the corrections are held fixed within each Newton step, the inner solve
retains its quadratic convergence, and with the corrections set to zero the
solver is byte-for-byte the classical drift-diffusion model.

## V. Results and Discussion

We run the identical device under four progressively-richer configurations —
(a) all-GaAs **homojunction**, Boltzmann, classical; (b) **+ heterojunction**
(GaAs/AlGaAs), Boltzmann, classical; (c) **+ Fermi–Dirac**; (d) **+
Schrödinger–Poisson** (the full four-physics model) — and record the equilibrium
electrostatics, the forward-bias I–V, and the optical output. Table II collects
the headline numbers.

**Table II. Per-physics ablation (300 nm × 10 nm DH-LED, 300 K).**

| Configuration | Built-in potential V_bi [V] | Peak IQE | Emission λ [nm] |
|---|:--:|:--:|:--:|
| Homojunction (Boltzmann, classical) | 1.389 | 0.901 | 873.1 |
| + Heterojunction | 1.750 | 0.908 | 873.1 |
| + Fermi–Dirac | 1.765 | 0.897 | 873.1 |
| + Schrödinger–Poisson (full) | 1.766 | 0.897 | 873.1 |

**A. Heterojunction electrostatics.** Introducing the AlGaAs cladding raises the
built-in potential by 0.36 V (1.389 → 1.750 V), the signature of the band-edge
offsets adding to the junction barrier. More importantly for a light emitter, the
band offsets confine carriers in the GaAs active region: Fig. 4 (left) shows the
equilibrium electron density, and Fig. 2 shows that the heterostructure reaches a
given IQE at roughly two orders of magnitude *lower* drive current than the
homojunction — the DH efficiency advantage, quantified. The I–V ablation (Fig. 3)
shows the corresponding turn-on shift.

**B. Fermi–Dirac degeneracy.** In the degenerately-doped cladding (doping ≈ N_c),
Boltzmann statistics over-count the carriers. Switching on Fermi–Dirac lowers the
peak IQE slightly (0.908 → 0.897) and marginally raises V_bi, the expected
degeneracy correction; the effect is confined to the cladding and vanishes in the
lightly-doped active region.

**C. Schrödinger–Poisson confinement.** In the 10 nm active layer the
self-consistent SP loop resolves four confined electron subbands, at
**E = [−1.031, −0.971, −0.789, −0.478] eV** (ascending, as required), and produces
a bounded quantum potential (Λ_n ∈ [−0.66, +0.10]). Physically, the correction
sets the carrier density peak *back* from the heterointerface and smooths the
classical profile — visible as the deviation of the green (full-model) curve from
the classical curves near the interface in Fig. 4 (left). Because the SP
correction conserves the per-slice charge, its effect on the *integrated*
efficiency of this particular (already strongly-confined) device is small (peak
IQE essentially unchanged), while it measurably redistributes the carriers — the
regime where SP corrections matter is exactly the few-nm active layers of scaled
emitters, and the method is in place for that study.

**D. Optical output.** The full model emits at **873.1 nm** (photon energy
1.420 eV), set by the GaAs active-region band gap, with a **peak IQE of 0.90**;
the IQE-vs-current curve (Fig. 2) rises through the injection regime and rolls
over as non-radiative Auger loss grows at high current, reproducing the
qualitative efficiency behaviour of a real LED. By construction the radiative
current equals qR_rad, so the light-emitting fraction of the terminal current is
recovered directly.

**Figures.**
- **Fig. 1** — `figures/fig1_iv.png`: total and radiative I–V of the full model.
- **Fig. 2** — `figures/fig2_iqe_ablation.png`: IQE vs. current, four-config
  ablation (the DH advantage).
- **Fig. 3** — `figures/fig3_iv_ablation.png`: I–V ablation (turn-on shift).
- **Fig. 4** — `figures/fig4_confinement.png`: equilibrium electron density and
  potential across the device (homo vs. hetero vs. full SP).
- **Fig. 5** — `figures/fig5_subbands.png`: the four confined electron subbands.

## VI. Validation

The solver is checked against analytic semiconductor physics by a unit-test suite
(≈ 110 tests): the built-in potential matches V_bi = V_T ln(N_A N_D / n_i²) for the
homojunction; mass action n·p = n_i² holds at equilibrium to machine precision;
the forward I–V is rectifying and exponential with ideality factor ≈ 1; and the
terminal currents at the two contacts are equal and opposite (conservation). The
recombination Jacobian derivatives match finite differences; the classical limit
(Λ = 0, Boltzmann) is byte-for-byte identical to the pure DD solver; the SP
subband eigenvalues come out ascending and the self-consistent loop converges with
a bounded quantum potential while conserving charge; and the optical read-out
gives IQE ∈ (0, 1] with the emission wavelength matching the active-region gap and
I_rad = qR_rad. These checks establish that the coupled four-physics result rests
on individually-verified components.

## VII. Conclusion and Future Work

We have demonstrated, in a single open, test-validated 2-D FEM drift-diffusion
solver, the fully-coupled combination of heterojunction electrostatics,
self-consistent Schrödinger–Poisson subband confinement, Fermi–Dirac degeneracy
and radiative recombination for a GaAs/AlGaAs double-heterostructure LED, and we
have attributed the change each effect produces through a controlled per-physics
ablation. The heterojunction raises the built-in potential and shifts the turn-on
so the DH device reaches a given IQE at ~100× lower current; Fermi–Dirac corrects
the degenerate-cladding over-count; the SP loop resolves the active-layer subbands
and redistributes the confined carriers; and the radiative channel yields a peak
IQE of 0.90 at 873 nm. The combination — open, reproducible, and coupling a
genuine self-consistent SP correction rather than a density-gradient surrogate —
fills a gap between the closed commercial TCAD tools and the existing open device
simulators.

Future work follows two directions already scoped in the solver's roadmap:
heterointerface thermionic-emission transport (for HBTs/HEMTs) and GaN
polarization charge (for nitride emitters), which would extend the same framework
to visible and UV LEDs; and, as a distinct study, adapting the transport and
recombination machinery — with field/density-dependent (EGDM) mobility and
Gauss–Fermi statistics — toward organic and quantum-dot LEDs.

## Reproducibility

All numbers and figures in this paper regenerate from the scripts in `code/`
(`ablation.py`, `make_figs.py`) against the open DDM.SPC solver; the raw ablation
data is in `data/ablation.json` and the figures in `figures/`.

## References

[1] Zh. I. Alferov, "Nobel Lecture: The double heterostructure concept and its
applications," *Rev. Mod. Phys.*, vol. 73, no. 3, pp. 767–782, 2001.

[2] J. E. Sanchez and R. Ananian-Cooper, "DEVSIM: A TCAD Semiconductor Device
Simulator," *Journal of Open Source Software*, vol. 7, no. 70, 3898, 2022.

[3] Cogenda Pte. Ltd., "Genius-TCAD-Open: Open-source edition of the Genius
Semiconductor Device Simulator." [Online]. Available:
https://github.com/cogenda/Genius-TCAD-Open

[4] Sandia National Laboratories, "Charon: A drift-diffusion / hydrodynamic
semiconductor device simulation code." [Online]. Available:
https://charon.sandia.gov

[5] "SEMIDV: A Compact Semiconductor Device Simulator with Quantum Effects,"
arXiv:2504.00214, 2025.

[6] C. de Falco, E. Gatti, A. L. Lacaita and R. Sacco, "Quantum-corrected
drift-diffusion models for transport in semiconductor devices" / A coupled
Schrödinger–drift-diffusion model for quantum semiconductor device simulations,
*J. Comput. Phys.*, vol. 176, pp. 149–166, 2002.

[7] A. De Mari, "An accurate numerical steady-state one-dimensional solution of
the p-n junction," *Solid-State Electronics*, vol. 11, pp. 33–58, 1968.

[8] D. Bednarczyk and J. Bednarczyk, "The approximation of the Fermi–Dirac
integral F_{1/2}(η)," *Physics Letters A*, vol. 64, no. 4, pp. 409–410, 1978.

[9] H. K. Gummel, "A self-consistent iterative scheme for one-dimensional steady
state transistor calculations," *IEEE Trans. Electron Devices*, vol. 11, pp.
455–465, 1964.

*Note on references [4], [6], [7], [8], [9]:* these are standard references for
the methods used; bibliographic details should be verified against the primary
sources before camera-ready submission.
