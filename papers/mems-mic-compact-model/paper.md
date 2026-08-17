# A Closed-Form Verilog-A Compact Model of a Capacitive MEMS Microphone for Transducer–CMOS Read-Out Co-Simulation

*Author Name(s) — Department / Institution (placeholders for camera-ready).*

## Abstract

Designing the analog front-end of a capacitive MEMS microphone requires
simulating the electromechanical transducer and its CMOS read-out *together*,
yet the finite-element (FE) models that capture the transducer physics cannot be
embedded in a transistor-level simulator, and existing lumped equivalent
circuits usually treat damping and the acoustic load with fitted parameters. We
present a physically-derived, closed-form compact model of a capacitive MEMS
microphone and its Verilog-A/SPICE realization. A single-mode Galerkin reduction
of the tensioned-membrane electromechanical PDE yields a lumped
electro-mechano-acoustic model in which, for the circular diaphragm, the modal
mass, stiffness, effective area *and* the gap capacitance
`C(q)=(εA/q)·ln[g₀/(g₀−q)]` are all closed-form. A first-principles squeeze-film
term (Škvor annular-cell Reynolds solution) makes the quality factor *derived*
not fitted; a back-cavity/vent network sets the low-frequency corner; and a
fluctuation–dissipation model yields the SNR. Verified against a full 2-D FE
reference across three designs, the compact model reproduces sensitivity to
within 0.8%, resonance to 1.5% and pull-in to 2%, while evaluating >3×10⁵ times
faster. Emitted as a SPICE subcircuit it matches the reference to 2×10⁻⁸, and we
demonstrate transducer + source-follower and transducer + charge-amplifier
co-simulations in ngspice.

## I. Introduction

Capacitive MEMS microphones are among the highest-volume microsystems shipped
today, and their performance is set as much by the CMOS read-out ASIC as by the
transducer: the bias network, the source-follower or charge-amplifier front-end
and their noise all shape the delivered sensitivity, bandwidth and SNR. This
demands **co-simulation** of the transducer with the read-out in one simulator.

FE models resolve the transducer physics but are distributed, nonlinear,
expensive, and cannot be instantiated in a SPICE/Spectre netlist. Lumped
equivalent circuits can, but published circuits commonly (i) fit the mechanical
quality factor instead of deriving squeeze-film damping, and (ii) fold the
acoustic load into empirical elements. This work closes that gap with a compact
model **derived end-to-end** from the governing equations, **closed-form** for the
circular diaphragm, delivered as a Verilog-A device plus a SPICE subcircuit.

**Contributions.** (1) A single-mode electro-mechano-acoustic compact model with
a closed-form, exactly-differentiable gap capacitance `C(q)` for the circular
mode, giving analytic sensitivity and pull-in. (2) A first-principles
squeeze-film damping term (Škvor), so Q and the thermal-mechanical noise (hence
SNR) follow from geometry and viscosity with no fitting. (3) A Verilog-A/SPICE
realization verified against full 2-D FEM (sensitivity <0.8%, resonance <1.5%,
pull-in <2%) at >3×10⁵ speed-up, with CMOS read-out co-simulation in ngspice.

**Capability comparison**

| Capability | Full FEM | Lumped EC (typ.) | This work |
|---|:--:|:--:|:--:|
| Distributed accuracy | ● | ○ | ◑ (<1% via modal reduction) |
| Runs inside SPICE/Spectre | ○ | ● | ● |
| Closed-form C(q), pull-in | ○ | ◑ | ● |
| *Derived* squeeze-film Q | ◑ | ○ | ● |
| Cavity/vent high-pass | ◑ | ◑ | ● |
| Thermal-mechanical SNR | ○ | ○ | ● |
| CMOS read-out co-simulation | ○ | ● | ● |

## II. Device and governing equations

A pre-tensioned circular diaphragm (radius `a`, tension per length `T`, areal
density `ρₛ=ρh`) faces a rigid perforated backplate across a rest gap `g₀` at
bias `V`. Its deflection `w` (positive toward the backplate) obeys

    ρₛ ẅ + b ẇ − T ∇²w = p + εV²/[2(g₀−w)²],   w|∂Ω = 0,

with `C = ε ∫ dA/(g₀−w)`. The mechanical part is a Poisson problem; its
linear-triangle FE discretization is the high-fidelity reference.

## III. Reduced-order compact model

### A. Modal (Galerkin) reduction

With `w(r,t)=q(t)φ(r)` and the fundamental (uniform-load) shape `φ`
(peak-normalized), projecting onto `φ` gives a 1-DOF oscillator

    m q̈ + c q̇ + k q = A_eff p + F_es(q,V),
    m = ∫ ρₛ φ² dA,   k = ∫ T |∇φ|² dA,   A_eff = ∫ φ dA.

For a clamped circular membrane the static shape is **exactly parabolic**,
`φ = 1 − (r/a)²`, so

    m = ρₛA/3,   k = 2πT,   A_eff = A/2,   ∫φ² dA = A/3,   A = πa².

The Rayleigh estimate `f₀ = (√6 / 2πa)·√(T/ρₛ)` is within 1.9% of the exact
Bessel value `2.4048 c_m/2πa`.

### B. Closed-form gap capacitance

Substituting `u = 1 − (r/a)²` the mode-integrated capacitance integrates exactly:

    C(q) = ε ∫ dA/(g₀ − qφ) = (εA/q)·ln[g₀/(g₀−q)],   C(0) = εA/g₀.

This single elementary expression is analytically differentiable to all orders,
so the electrostatic force, transduction and pull-in are all closed-form. Its
limits `C'(0)=εA/2g₀²` and `C''(0)=2εA/3g₀³` match the general integrals
`ε∫φ dA/g₀²` and `2ε∫φ² dA/g₀³`.

### C. Electrostatic force, transduction, pull-in

`F_es = ½V²C'(q)`. Linearizing `Q=C(q)V` about `q₀` gives the transduction
factor `Γ = V_B C'(q₀)`, the constant-charge output `v_oc = (Γ/C₀)q`, and the
spring softening `k_es = ½V_B²C''(q₀)`. Pull-in eliminates `V` from the
equilibrium and marginal-stability conditions:

    q_PI = C'(q_PI)/C''(q_PI),   V_PI = √[2k / C''(q_PI)].

For a rigid plate (`φ≡1`) this reduces to `q_PI = g₀/3`,
`V_PI = √(8k g₀³ / 27εA)`.

### D. Squeeze-film damping (derived Q)

Each hole vents one cell (radius `r_c`, `πr_c²=A/N`) with a central hole radius
`r₀`. The incompressible linearized Reynolds equation
`(g₀³/12μ) r⁻¹(rP')' = u` with `P(r₀)=0`, `P'(r_c)=0` integrates to a per-cell
damping force `F = −b_cell u`:

    b_cell = (3πμ/2g₀³) r_c⁴ K(β),   K(β) = 4β² − β⁴ − 4 ln β − 3,   β = r₀/r_c,

(Škvor attenuation). Then `b_area = b_cell/πr_c²`, and the modal damping is
`c = b_area ∫φ² dA`. The 1/g₀³ scaling and `K→0` as `β→1` are recovered;
`Q = √(mk)/c` is **derived** from geometry and viscosity.

### E. Back-cavity and vent acoustics

The back cavity is an acoustic compliance `C_ab = V_b/ρ₀c₀²`, stiffening the
mode by `k_cav = A_eff²/C_ab`. A pressure-equalization vent
(`R_av = 8μℓ_v/πa_v⁴`, mass `M_av`) shorts the diaphragm at low frequency. The
two-node acoustic network gives

    (q/p)(ω) = A_eff(1 − p̂_b)/Z_m,   Z_m = k_net − ω²m + jωc,
    p̂_b = G/(G + jωC_ab),   G = jωA_eff²/Z_m + 1/Z_vent,   Z_vent = R_av + jωM_av,

with `k_net = k − k_es`. At DC the vent equalizes (`q→0`); the high-pass corner
is `f_hp = 1/2πR_av C_ab`. Mid-band, `q/p = A_eff/(k_net + k_cav)`. Output
sensitivity `S(ω) = (Γ/C₀)|q/p|`.

### F. Thermal-mechanical noise and SNR

By fluctuation–dissipation the squeeze-film damping radiates force noise
`S_F = 4k_BTc`; input-referred `S_p = 4k_BTc/A_eff²` (white). A-weighted
band-integration gives the EIN, and `SNR = 94 − EIN_dBA` (ref 1 Pa). This is the
mechanical-thermal **bound**; real parts are lower once read-out electrical noise
is added.

## IV. Verilog-A / SPICE realization

The equations map one-to-one to a Verilog-A module (`mems_mic.va`): electrical
capacitor terminals, a pressure input node, and internal nodes for modal
displacement/velocity, cavity pressure and the vent state. The mechanical KCL
realizes the modal ODE, the electrical branch contributes `i = d[C(q)V]/dt` with
the closed-form `C(q)`, and the cavity/vent nodes realize the acoustic network.
For fast small-signal work an equivalent SPICE subcircuit is emitted: a
force-driven mechanical R-L-C (`L_m=m`, `R_m=c`, `C_m=1/k_net`) whose loop
current is the diaphragm velocity, coupled by a current-controlled source of gain
`Γ` into the electrical port capacitance `C₀`.

## V. Results

Reference design: 1 mm-radius, 0.5 µm Si₃N₄ diaphragm, `T=6` N/m, `g₀=6` µm,
`V_B=10` V, `V_b=10` mm³. Operating point: `C₀=4.64` pF, `m=1.62` nkg, `k=37.7`
N/m, `k_cav=35.0` N/m, `Q=6.2`, `V_PI=15.6` V, `f₀=32.4` kHz, `f_hp=0.13` Hz,
sensitivity 21.2 mV/Pa (−33.5 dBV/Pa), mechanical-thermal SNR 83 dB(A).

### A. Accuracy vs. full FEM (bare diaphragm — isolates modal-reduction error)

| Design | Sensitivity | f₀ | V_PI | Speed-up |
|---|--:|--:|--:|--:|
| D1 (0.8 mm, T4) | 0.74% | 1.5% | 2.0% | 3.6×10⁵ |
| D2 (1.0 mm, T6) | 0.77% | 1.5% | 2.0% | 5.5×10⁵ |
| D3 (1.2 mm, T8) | 0.78% | 1.5% | 2.0% | 4.8×10⁵ |

The closed-form `C(q)` matches the FE capacitance at a finite deflection to 0.3%.
(Fig. 1.)

### B. Speed

Each FE figure-of-merit (a nonlinear biased solve plus a pull-in continuation)
costs tens of seconds on a ~1800-node mesh; the closed-form compact evaluation
costs ~0.1 ms — a >3×10⁵ speed-up (Fig. 3), making design-space exploration and
in-the-loop circuit simulation practical.

### C. Read-out co-simulation

The SPICE macromodel reproduces the analytic transducer to 2×10⁻⁸ in the
mid-band. ngspice then co-simulates it with (i) an NMOS source-follower + 10 GΩ
bias resistor and (ii) an op-amp charge amplifier, exposing buffering, the
electrical high-pass, and the `C₀/C_f` charge gain that FEM cannot show (Fig. 2).

### D. Damping, noise, distortion

Fig. 4 shows the derived `Q(g₀)` and `K(β)`; Fig. 5 maps SNR against radius and
gap with the commercial band; Fig. 6 gives THD vs. SPL from the nonlinear
transient.

### E. Commercial anchors

Fig. 7 places the compact-model design cloud against four shipping analog MEMS
microphones. Predicted sensitivities (−33 to −45 dBV/Pa) overlap the commercial
range; the model's SNR sits above the parts because it reports the
mechanical-thermal bound — the gap being read-out electrical noise captured in
co-simulation.

## VI. Conclusion

A physically-derived, largely closed-form compact model of a capacitive MEMS
microphone and its Verilog-A/SPICE realization tracks a full FE reference to <2%
at >3×10⁵ speed-up and, unlike FEM, co-simulates with a CMOS front-end. The
circular-mode `C(q)=(εA/q)ln[g₀/(g₀−q)]` makes sensitivity and pull-in analytic;
a Škvor squeeze-film term makes Q and the SNR predictive rather than fitted; and
a cavity/vent network sets the audio band. Future work: non-circular and
dual-backplate diaphragms, the squeeze-film gas-spring at high squeeze number,
and calibration against measured devices.

## References

1. R. Škvor, "On the acoustical resistance due to viscous losses in the air gap
   of electrostatic transducers," *Acustica*, 19:295–299, 1967.
2. S. D. Senturia, *Microsystem Design*. Kluwer, 2001.
3. J. B. Starr, "Squeeze-film damping in solid-state accelerometers," *IEEE
   Solid-State Sensor and Actuator Workshop*, 1990.
4. M. Bao, *Analysis and Design Principles of MEMS Devices*. Elsevier, 2005.
5. C. Zuo et al., "Lumped-element modeling of capacitive MEMS microphones,"
   *J. Microelectromech. Syst.*, 2019.
6. Accellera, *Verilog-AMS Language Reference Manual*, v2.4.0, 2014.
7. H. Vogt et al., *ngspice, the open-source SPICE circuit simulator*.
