COMPLETE WORKFLOW: Adsorption Enthalpy of n-heptane on IRMOF-13 at Infinite Dilution
=====================================================================================

OBJECTIVE:
Determine the adsorption enthalpy of n-heptane on IRMOF-13 using RASPA simulation at infinite dilution conditions.

COMPLETE WORKFLOW SETUP:

=== STEP 1: PREREQUISITE SIMULATION (simulation_2/) ===
Helium Void Fraction Calculation - MUST BE RUN FIRST

Files created in simulation_2/:
- framework.cif: IRMOF-13 crystal structure
- helium.def: helium molecule definition
- force_field.def: force field parameters
- force_field_mixing_rules.def: mixing rules
- pseudo_atoms.def: atomic properties
- helium_void_fraction.input: helium void fraction simulation input

Simulation settings:
- Monte Carlo simulation
- 1000 cycles (reduced for speed)
- Temperature: 298.0 K
- Pressure: 1e5 Pa
- Unit cells: [2, 2, 1]
- Properties computed: HeliumVoidFraction, PoreVolume, SurfaceArea

=== STEP 2: MAIN SIMULATION (simulation_1/) ===
Adsorption Enthalpy Calculation at Infinite Dilution

Files created in simulation_1/:
- framework.cif: IRMOF-13 crystal structure
- n-heptane.def: n-heptane molecule definition
- helium.def: helium molecule definition (for reference)
- force_field.def: force field parameters
- force_field_mixing_rules.def: mixing rules
- pseudo_atoms.def: atomic properties
- simulation.input: main adsorption enthalpy simulation
- readme.txt: detailed simulation setup explanation

Simulation settings:
- Monte Carlo simulation (framework-based)
- 1000 cycles (reduced for speed as requested)
- Temperature: 298.0 K
- Pressure: 1e5 Pa
- Unit cells: [2, 2, 1]
- Infinite dilution: CreateNumberOfMolecules = 0 (insertion/deletion moves)
- Properties computed: HeatOfAdsorption, HenryCoefficient, NumberOfMolecules, EnergyHistogram

EXECUTION ORDER:
1. Run helium void fraction simulation in simulation_2/ first
2. Extract helium void fraction value from output
3. Update simulation_1/simulation.input with calculated void fraction
4. Run main adsorption enthalpy simulation in simulation_1/
5. Parse output files to extract adsorption enthalpy values

KEY PARAMETERS:
- Framework: IRMOF-13 (Metal-Organic Framework)
- Adsorbate: n-heptane (C7H16)
- Conditions: Infinite dilution at 298 K and 1 bar
- Cutoff: 12.8 Angstrom for both VDW and Coulomb interactions
- Reduced cycles: 1000 (normally 10,000-100,000 for production)

EXPECTED OUTPUTS:
- Helium void fraction value (typically 0.8-0.9 for MOFs)
- Adsorption enthalpy of n-heptane (kJ/mol)
- Henry coefficient at infinite dilution
- Statistical uncertainties for all calculated properties

NOTE: All simulations use reduced cycle counts (1/10 of typical) for faster execution as requested.
For production-quality results, increase cycles to 10,000-100,000.
