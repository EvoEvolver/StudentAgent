ADSORPTION ENTHALPY CALCULATION: n-heptane on IRMOF-13 at Infinite Dilution
=============================================================================

OBJECTIVE:
Determine the adsorption enthalpy of n-heptane on IRMOF-13 using RASPA simulation at infinite dilution conditions.

COMPLETE SETUP ACCOMPLISHED:

=== FILES CREATED ===
1. framework.cif - IRMOF-13 crystal structure (unit cells: 2x2x1)
2. n-heptane.def - n-heptane molecule definition
3. helium.def - helium molecule definition
4. force_field.def - force field parameters
5. force_field_mixing_rules.def - mixing rules
6. pseudo_atoms.def - atomic properties
7. simulation.input - main adsorption enthalpy simulation
8. helium_void_fraction.input - prerequisite helium void fraction simulation

=== EXECUTION WORKFLOW ===

STEP 1: Run Prerequisite Simulation
- Execute: helium_void_fraction.input
- Purpose: Calculate helium void fraction (required for accurate enthalpy)
- Expected output: Void fraction value (~0.8-0.9 for MOFs)

STEP 2: Update Main Simulation
- Update simulation.input with calculated void fraction
- Replace assumed value (0.85) with actual calculated value

STEP 3: Run Main Simulation
- Execute: simulation.input
- Purpose: Calculate adsorption enthalpy at infinite dilution
- Expected outputs: Heat of adsorption, Henry coefficient

=== SIMULATION PARAMETERS ===
- Framework: IRMOF-13 (Metal-Organic Framework)
- Adsorbate: n-heptane (C7H16)
- Temperature: 298.0 K (25°C)
- Pressure: 1e5 Pa (1 bar)
- Simulation type: Monte Carlo
- Cycles: 1000 (reduced for speed as requested)
- Cutoffs: 12.8 Å (VDW and Coulomb)
- Infinite dilution: CreateNumberOfMolecules = 0

=== KEY PROPERTIES COMPUTED ===
Helium simulation:
- HeliumVoidFraction
- PoreVolume
- SurfaceArea

Main simulation:
- HeatOfAdsorption (primary objective)
- HenryCoefficient
- NumberOfMolecules
- EnergyHistogram

=== IMPORTANT NOTES ===
1. ALWAYS run helium void fraction simulation FIRST
2. Use calculated void fraction in main simulation
3. Cycles reduced to 1000 (1/10 of typical) for speed
4. For production: use 10,000-100,000 cycles
5. Results will include statistical uncertainties

=== EXPECTED RESULTS ===
- Adsorption enthalpy of n-heptane on IRMOF-13 (kJ/mol)
- Henry coefficient at infinite dilution
- Framework characterization (void fraction, surface area, pore volume)

All prerequisite steps completed. Ready for simulation execution.
