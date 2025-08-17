HENRY COEFFICIENT SIMULATION SETUP FOR N2 AND METHANE ON IRMOF-13
==================================================================

OBJECTIVE:
Determine the Henry coefficient of nitrogen (N2) and methane on IRMOF-13 framework using RASPA simulation.

STEPS COMPLETED:

1. FRAMEWORK SETUP:
   - Loaded IRMOF-13 framework using framework loader
   - Generated framework.cif file
   - Unit cells set to [2, 2, 1] as recommended for 12.8 Å cutoff
   - Used provided helium void fraction: 0.877

2. MOLECULE SETUP:
   - Loaded nitrogen molecule (nitrogen.def)
   - Loaded methane molecule (methane.def)
   - Generated corresponding force field files

3. SIMULATION CONFIGURATION:
   - SimulationType: MonteCarlo (required for Henry coefficient)
   - Method: Widom insertion (WidomProbability 1.0)
   - CreateNumberOfMolecules: 0 (no actual insertion, energy calculation only)
   - Temperature: 298 K (standard conditions)
   - Cycles: 5000 (reduced as instructed, 1/10 of typical values)
   - Initialization: 500 cycles

4. KEY PARAMETERS:
   - Forcefield: local
   - CutOff: 12.8 Å (VDW and Coulomb)
   - Charge method: Ewald with 1e-6 precision
   - IdealGasRosenbluthWeight: 1.0 (estimated, should be calculated separately)

IMPORTANT NOTES:
- This simulation setup is ready but NOT executed
- IdealGasRosenbluthWeight values (1.0) are estimates and should ideally be calculated in separate prerequisite simulations
- The simulation will output Henry coefficients in [mol/kg/Pa] units
- Both components (N2 and methane) will be calculated simultaneously

FILES GENERATED:
- simulation.input (main simulation file)
- framework.cif (IRMOF-13 structure)
- nitrogen.def (nitrogen molecule definition)
- methane.def (methane molecule definition)
- force_field.def (force field parameters)
- pseudo_atoms.def (atomic parameters)
- force_field_mixing_rules.def (mixing rules)

TO RUN:
Execute RASPA with the simulation.input file to obtain Henry coefficients for both molecules.