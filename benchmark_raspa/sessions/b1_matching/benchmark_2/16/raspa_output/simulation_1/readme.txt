RASPA Henry Coefficient Simulation Setup for n-pentane and n-heptane on IRMOF-13
================================================================================

OVERVIEW:
This simulation setup calculates the Henry coefficients of n-pentane and n-heptane
in the IRMOF-13 metal-organic framework using the Widom insertion method.

FILES GENERATED:
1. framework.cif - IRMOF-13 crystal structure definition
2. pentane.def - n-pentane molecule definition and geometry
3. n-heptane.def - n-heptane molecule definition and geometry
4. force_field.def - Force field parameters for interactions
5. force_field_mixing_rules.def - Mixing rules for force field
6. pseudo_atoms.def - Pseudoatom definitions
7. simulation.input - Main RASPA input file
8. readme.txt - This documentation file

SETUP STEPS PERFORMED:
1. Loaded IRMOF-13 framework using framework loader
   - Generated framework.cif with unit cells [2, 2, 1] for 12.8 Å cutoff

2. Loaded molecule definitions for n-pentane and n-heptane
   - Generated .def files with molecular geometry and force field parameters

3. Created simulation.input file with Henry coefficient calculation parameters:
   - Simulation Type: Monte Carlo
   - Method: Widom insertion (WidomProbability = 1.0)
   - No actual molecules inserted (CreateNumberOfMolecules = 0)
   - Temperature: 298.0 K
   - Pressure: 1e5 Pa (1 bar)
   - Cycles: 100,000 (with 10,000 initialization cycles)

KEY INPUT PARAMETERS PROVIDED:
- Helium void fraction: 0.877
- Ideal gas Rosenbluth weight for n-pentane: 0.0197439
- Ideal gas Rosenbluth weight for n-heptane: 0.0004450

SIMULATION DETAILS:
- Framework: IRMOF-13 with unit cells [2, 2, 1]
- Cutoff distances: 12.8 Å for both VDW and Coulomb interactions
- Charge method: Ewald summation with 1e-6 precision
- Force field: Local (using generated force field files)

TO RUN THE SIMULATION:
Execute RASPA with the simulation.input file. The simulation will calculate
Henry coefficients for both components simultaneously using Widom insertions.

EXPECTED OUTPUT:
Henry coefficients in [mol/kg/Pa] units for both n-pentane and n-heptane,
along with statistical uncertainties.

NOTE:
This setup is ready for execution but has not been run yet as requested.
The provided Rosenbluth weights eliminate the need for prerequisite simulations.