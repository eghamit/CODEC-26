# Applications Exploration — DDM.SPC Solver

*What can this drift-diffusion + Schrödinger–Poisson device solver actually
simulate, across biomedical detection, sensors, multiphysics, and
microelectronics — and with which (including novel) materials?*

This document maps the **real capability envelope** of `DDM_SPC` onto four
application domains, rates each idea by how much work it needs today, and calls
out the physics that would have to be added for the rest. It is deliberately
honest about limits: an idea marked *"needs a new term"* is flagged as such
rather than sold as ready.

---

## 1. What the solver actually is (capability envelope)

`DDM.SPC` is a **2-D, steady-state, finite-element drift-diffusion** semiconductor
device solver in quasi-Fermi variables, fully-coupled Newton with the exact
analytic Jacobian. On top of the classical core it has:

| Capability | Status | Enables |
|---|---|---|
| Coupled Poisson + electron/hole continuity | ✅ core | any diode/transistor DC operating point |
| Per-region materials + **heterojunctions** (band offsets, ε(x), μ(x)) | ✅ Phase 2A | HBTs, LEDs, HEM-like stacks, graded gaps |
| **SRH + Auger** non-radiative recombination | ✅ | leakage, lifetime, dark current |
| **Radiative** recombination + **optical output** (rate, IQE, wavelength) | ✅ | LEDs, light emission |
| **Schrödinger–Poisson** quantum-confinement correction | ✅ | thin-body/quantum-well threshold & density shifts |
| **MOS gate** (insulated, work-function referenced) | ✅ Phase 2C | MOSFETs, MOS caps, **ISFET/BioFET front-ends** |
| **Schottky** contact (barrier height) | ✅ Phase 2C | MESFETs, Schottky diodes, **Schottky gas sensors** |
| **Fermi–Dirac** statistics (degeneracy shift) | ✅ Phase 2B | degenerate S/D, 2DEG, heavy doping |
| Terminal current + local current density (SI amperes) | ✅ | I–V, transfer/output curves |

### Hard limits (be honest about these)

These are **not** in the solver today; anything relying on them is a *roadmap*
item, not a ready simulation:

- **Steady-state only.** No transient (`d/dt`), no AC small-signal, no
  C–V/impedance/noise as a native output. (C–V-*style* curves can be built by
  sweeping bias and reading charge/potential, as the biosensor example does.)
- **No optical *generation*/absorption.** The optical model *emits* (radiative
  recombination); there is no photogeneration source term, so **photodiodes,
  solar cells, image sensors and photodetector biosensors need a generation
  term added** (see §6, extension G1).
- **No self-heating / heat-transport equation.** "Multiphysics" today means the
  *electro-quantum-optical* coupling that is already in the code, not thermal,
  mechanical, piezo, or fluidic physics (see §4).
- **No polarization charge** (spontaneous/piezo) — flagged in the README as a
  future item; GaN/AlGaN HEMTs are only partially representable without it.
- **No ionized-impurity/field-dependent mobility models** beyond constant
  per-region μ; velocity saturation and high-field effects are not modelled.
- **2-D, per-unit-depth** (set `device_depth` for absolute amperes).

Keeping these in view, here is what each domain can do.

---

## 2. Biomedical devices for disease detection

The dominant electronic biosensor is the **field-effect biosensor**: a
MOSFET-like device whose gate is functionalised so that binding of a biomarker
(DNA, antigen, ion, metabolite) changes the surface potential and hence the
channel current. That surface-potential change is **precisely the gate
work-function term** in `GateContact` (`u_gate = V/VT + (Phi_ref − Phi_m)/VT`),
so this whole class is representable *today*.

| Device idea | Biomarker / use | How it maps to DDM.SPC | Feasibility |
|---|---|---|---|
| **ISFET (pH sensor)** | blood/saliva pH, metabolic panels | MOS cap/FET; analyte→ΔΦ on the gate → threshold shift | ✅ **works today** — see `examples/biosensor_isfet.py` |
| **BioFET / ImmunoFET** | antigen–antibody, cardiac troponin, CRP | same surface-potential transduction, different ΔΦ mapping | ✅ works today (as ISFET front-end) |
| **DNA-FET / genFET** | DNA hybridisation, pathogen DNA/RNA | hybridisation charge → gate ΔΦ; sub-threshold slope = sensitivity | ✅ works today |
| **Glucose / enzymatic FET** | diabetes management | enzymatic reaction → local pH/charge → gate ΔΦ | ✅ works today |
| **Nanowire/thin-body BioFET** | ultra-sensitive, few-molecule | thin body ⇒ turn on `quantum="schrodinger"` for the confinement shift | ✅ works today (quantum-corrected) |
| **Extended-gate / dual-gate ISFET** | drift-compensated readout | two gate contacts, differential threshold | ✅ works today |
| **Photonic biosensor / fluorescence detector** | labelled assays, lab-on-chip optics | needs a **photogeneration** term (extension G1) | 🔶 needs new term |
| **Wide-bandgap ISFET (GaN/SiC)** | harsh-media, implantable, autoclavable | swap region material to `gan`/`sic`; robust surface | ✅ works today (materials in library) |

**Worked example.** `examples/biosensor_isfet.py` builds the ISFET MOS-capacitor
front-end on the built-in structured mesh, sweeps the gate through
accumulation→depletion→inversion for a clean surface and for one carrying bound
analyte (a +0.20 eV surface step), and extracts the **threshold shift** — the
sensor's actual signal. It reports a +200 mV response with the expected
near-unity coupling. The **sub-threshold slope** and the **ΔVth-per-ΔΦ
sensitivity** are the figures of merit a real assay would be characterised on,
and both come straight out of the sweep.

**Key idea:** any surface-binding assay whose end effect is a **charge or dipole
at the oxide surface** is an in-scope `ΔΦ_gate` study today. Assays whose
readout is **optical** (fluorescence, absorbance) or **transient** (binding
kinetics) need extensions G1 / T1.

---

## 3. Sensors

Beyond biomedical, the FET/diode sensing family maps well; the split is again
"surface-potential or barrier modulation" (ready) vs. "photogeneration or
transient" (needs a term).

| Sensor | Transduction | DDM.SPC mapping | Feasibility |
|---|---|---|---|
| **Chemical / gas FET (ChemFET)** | adsorbate dipole → ΔΦ_gate | gate work-function sweep (as ISFET) | ✅ works today |
| **Catalytic-gate H₂ sensor (Pd-gate)** | H dissolves in Pd → ΔΦ | gate work-function shift | ✅ works today |
| **Schottky-diode gas sensor** | adsorbate → barrier height Φ_Bn | `schottky` contact, sweep `barrier` | ✅ works today |
| **Ion / heavy-metal sensor** | ISE membrane potential → ΔΦ | gate work-function shift | ✅ works today |
| **Pressure/strain (piezoresistive)** | strain → mobility/gap change | edit region μ or Eg by hand (no native strain) | 🔶 partial / manual |
| **Piezoelectric (GaN/AlGaN) sensor** | strain → polarization charge | needs **polarization-charge** term | 🔶 needs new term |
| **Photodetector / photodiode** | light → e-h pairs → photocurrent | needs **photogeneration** term (G1) | 🔶 needs new term |
| **Solar cell / photovoltaic** | absorption → carrier collection | needs G1 (generation) + spectral model | 🔶 needs new term |
| **APD / UV detector (SiC, GaN, AlGaN)** | high-field multiplication | needs G1 **and** impact-ionization term | 🔴 needs two terms |
| **Temperature sensor (diode Vf(T))** | I–V shift with T | set `temperature` per material file; DC only | ✅ works today (parametric in T) |
| **Magnetic / Hall sensor** | Lorentz deflection | needs magnetic-field transport term | 🔴 out of scope |

**Ready-today sensor studies** cluster tightly around two knobs the solver
already exposes: the **gate work function** (`GateContact`) and the **Schottky
barrier** (`SchottkyContact`). Any sensor whose primitive is "adsorption/binding
moves a surface potential or a barrier" is a bias sweep away. The temperature
knob (`temperature` in the material file) also lets you produce the classic
diode-thermometer Vf(T) family as a parametric set of DC solves.

---

## 4. Multiphysics simulation

**Framing matters here.** "Multiphysics" in the full COMSOL/TCAD sense (electro-
thermal-mechanical-fluidic) is **not** what this solver is — and claiming
otherwise would be wrong. What it *does* have is a genuine, self-consistent
**electro-quantum-optical multiphysics core**:

- **Electrostatics ⟷ carrier transport** — Poisson coupled to both continuity
  equations, solved monolithically (not Gummel-split) with the exact Jacobian.
- **Transport ⟷ quantum mechanics** — the Schrödinger–Poisson outer loop feeds a
  1-D confinement solution back into the 2-D drift-diffusion density exponent
  and re-converges. This *is* two coupled physics (semiclassical transport +
  quantum confinement) exchanging fields self-consistently.
- **Transport ⟷ optics** — radiative recombination converts the converged
  electrical state into photon rate / IQE / wavelength, consistently with the
  same densities and quadrature the electrical solve used.
- **Material heterogeneity ⟷ electrostatics** — position-dependent ε(x), μ(x)
  and band offsets couple region materials into one electrostatic problem.

So the honest multiphysics story is: **already coupled** = {electrostatics,
drift-diffusion, quantum confinement, optical emission, heterojunction band
engineering}. Good demonstrator ideas that exercise this real coupling:

| Multiphysics demo | Coupled physics exercised | Feasibility |
|---|---|---|
| Quantum-corrected MOS threshold vs. body thickness | electrostatics + confinement | ✅ works today |
| Heterojunction LED with QW active region + IQE | transport + hetero + optics + (quantum) | ✅ works today (`examples/led_heterojunction_quantum.py`) |
| Degenerate 2DEG channel (Fermi–Dirac + confinement) | statistics + confinement + transport | ✅ works today |
| Graded-gap / bandgap-engineered drift field | hetero + transport | ✅ works today |

### Roadmap couplings (what "full" multiphysics would add)

| Added physics | Unlocks | Effort |
|---|---|---|
| **Electro-thermal** (lattice heat equation + T-dependent μ, ni) | self-heating, power-device SOA, thermal sensors | 🔴 major (new PDE + coupling loop) |
| **Piezo/spontaneous polarization** | GaN/AlGaN HEMT 2DEG, piezo sensors, SAW | 🔶 moderate (fixed interface charge + fields) |
| **Opto-electronic generation** (absorption) | photodetectors, solar, image sensors | 🔶 moderate (source term G1) |
| **Transient / AC** | dynamics, C–V, RF, noise, binding kinetics | 🔴 major (time integration / linearized AC) |

The architecture is friendly to these: the outer self-consistency loop that
already wraps Newton for Schrödinger–Poisson and Fermi–Dirac is the natural
place to hang an electro-thermal or polarization loop.

---

## 5. Microelectronics & integrated circuits

This is the solver's **home turf** — the classical drift-diffusion device
physics IC design rests on. Most of it works today with the bundled meshes/materials.

| Device | Mapping | Feasibility |
|---|---|---|
| **PN / PIN diode** | core | ✅ (`examples/pn_diode_2d.py`, `run_pin_diode.py`) |
| **Schottky diode / rectifier** | `schottky` contact | ✅ works today |
| **BJT / HBT** | 3-region + hetero for HBT | ✅ meshes present (`meshing/bjt_*`), `examples/run_bjt*.py` |
| **MOSFET (n-channel)** | MOS gate + n+ S/D | ✅ (`examples/mosfet_v3.py`, tuned gmsh mesh) |
| **MESFET** | `schottky` gate on channel | ✅ works today |
| **CMOS inverter / logic cell** | two MOSFETs; DC transfer curve | 🔶 buildable (multi-device mesh; DC only, no switching) |
| **Power diode / MOSFET (SiC, GaN)** | wide-gap materials + drift region | 🔶 DC blocking/on-state OK; **no impact-ionization → no true breakdown** |
| **HEMT (AlGaN/GaN)** | hetero 2DEG | 🔶 partial — **needs polarization charge** for the real 2DEG |
| **Quantum-well / thin-body FET, FinFET slice** | confinement correction | ✅ works today (`quantum="schrodinger"`) |
| **Tunnel diode / TFET** | band-to-band tunneling | 🔴 needs a tunneling generation term |
| **Memory (DRAM cell leakage, floating-gate)** | electrostatics + leakage | 🔶 static leakage OK; retention/programming need transient |
| **TFT (a-Si/oxide/organic display driver)** | add a TFT material file | 🔶 buildable with a new material file |

**IC-relevant studies ready today:** threshold-voltage engineering (body
doping, gate work function, quantum correction), short-channel electrostatics in
2-D, source/drain degeneracy with Fermi–Dirac, heterojunction band engineering
for HBTs and drift fields, and leakage/ideality from SRH+Auger. What IC work
needs the roadmap: **switching/timing** (transient), **breakdown** (impact
ionization), **RF/C–V** (AC), and **self-heating** (electro-thermal).

---

## 6. Novel / "noble" materials

*Read "noble material" as **novel materials** — wide-bandgap, compound, and
emerging semiconductors.* The material system is a simple, extensible text-file
format (`DDM_SPC/materials/*.txt`); adding a material is one file. **Bundled
today:** `silicon`, `germanium`, `gaas`, `algaas`, `gan`, `algan`, `gap`,
`sic`, `sio2`.

| Material | In library? | Standout use | Notes |
|---|---|---|---|
| **SiC (4H)** | ✅ `sic` | power, harsh-environment, UV | wide gap, high field; no impact-ionization yet |
| **GaN / AlGaN** | ✅ `gan`/`algan` | HEMTs, UV LEDs, power, piezo sensors | polarization charge still to come |
| **GaAs / AlGaAs** | ✅ | HBTs, IR LEDs, RF | direct-gap `B` present for LEDs |
| **GaP** | ✅ | visible LEDs (indirect) | radiative guardrail warns (indirect) |
| **Ge / SiGe** | ✅ `germanium` | strained-Si, IR, hetero | SiGe = add a graded file |
| **Ga₂O₃ (ultra-wide-gap)** | ➕ add file | ~4.8 eV power/UV-solar-blind | needs a new material file |
| **Diamond** | ➕ add file | ultimate power/thermal, UV | needs a new material file |
| **InGaAs / InP / InAlAs** | ➕ add file | telecom, HEMT, IR detectors | new files; direct-gap `B` for LEDs |
| **2-D (MoS₂, WSe₂, graphene)** | ➕ add file | thin-body FETs, ultra-sens. biosensors | effective-mass files; graphene gapless ⇒ care |
| **Perovskite / organic** | ➕ add file | flexible PV, display, sensors | new files; PV needs generation term G1 |

**To add a novel material:** copy an existing `.txt`, set `epsilon_r`,
`electron_affinity`, `band_gap`, `ni`, `Nc`, `Nv`, mobilities, `m_eff_n/p`, and
(for direct-gap emitters) `B`; load it by name via `region_properties` or
`material=`. Wide-gap materials plug straight into the existing MOS/Schottky/LED
flows — a **SiC or GaN ISFET** for harsh-media/implantable biosensing, for
instance, is just a material swap on the §2 example.

### Extension hooks referenced above

- **G1 — Photogeneration/absorption source term.** Add `G(x)` to the two
  continuity RHS (mirrors the existing recombination term's assembly path). This
  single addition unlocks **photodiodes, solar cells, image sensors, and optical
  biosensors** across all four domains — the highest-leverage next feature.
- **T1 — Transient/AC.** Time-derivative (or linearized AC) on the continuity
  equations for dynamics, C–V, RF, and binding kinetics.
- **P1 — Polarization charge.** Fixed spontaneous/piezo interface charge for
  real GaN/AlGaN HEMTs and piezoelectric sensors.
- **Th1 — Electro-thermal.** Lattice heat equation coupled through the existing
  outer self-consistency loop for self-heating and thermal sensors.

---

## 7. Feasibility summary

**Works today (bias-sweep or material swap):** ISFET/BioFET/DNA-FET/glucose
biosensors, ChemFET/Schottky gas sensors, ion sensors, diode thermometers,
PN/PIN/Schottky diodes, BJT/HBT, MOSFET/MESFET, quantum-corrected thin-body
FETs, heterojunction LEDs with IQE/wavelength, degenerate 2DEG channels,
electro-quantum-optical multiphysics demos, and all of the above on the bundled
**wide-bandgap** materials (SiC, GaN/AlGaN, GaAs/AlGaAs, GaP, Ge).

**One term away (highest leverage first):** photodetectors / solar / image
sensors / optical biosensors → **G1 (photogeneration)**; GaN HEMTs & piezo
sensors → **P1 (polarization)**; dynamics / C–V / RF / kinetics → **T1
(transient/AC)**; self-heating & power SOA → **Th1 (electro-thermal)**;
breakdown/APD → impact-ionization; TFET → tunneling.

**Out of scope:** magnetic/Hall, fluidic, and full mechanical/acoustic physics.

---

### Deliverable in this exploration

`examples/biosensor_isfet.py` — a runnable, self-contained **ISFET/BioFET
biosensor front-end** (no external mesher). It demonstrates domains **1
(biomedical)** and **2 (sensors)** on the real solver: analyte binding →
gate-work-function shift → extracted threshold shift (the sensor signal), with
near-unity coupling as expected. It is the template for every surface-potential
biosensor and chemical-sensor study listed above.
