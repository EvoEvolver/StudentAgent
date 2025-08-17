RASPA Simulation Setup: Adsorption Enthalpy of n-pentane on IRMOF-13 at Infinite Dilution
======================================================================================

Objective:
Determine the adsorption enthalpy of n-pentane on IRMOF-13 using Monte Carlo simulation at infinite dilution conditions.

Given Parameters:
- Helium void fraction: 0.877
- Ideal gas Rosenbluth weight for n-pentane: 0.0197439
- Framework: IRMOF-13
- Molecule: n-pentane

Simulation Setup Steps:

1. Framework Loading:
   - Loaded IRMOF-13 framework as framework.cif
   - Unit cells set to [2, 2, 1] based on cutoff requirements (12.8 Å)

2. Molecule Loading:
   - Generated n-pentane molecule definition files (pentane.def)
   - Generated corresponding force field files

3. Input File Configuration:
   - Simulation Type: Monte Carlo
   - Cycles: 2500 (reduced from typical 25000 for faster execution)
   - Initialization Cycles: 1000
   - Temperature: 300.0 K
   - Pressure: 0.0 Pa (infinite dilution condition)
   - CreateNumberOfMolecules: 1 (for infinite dilution enthalpy calculation)
   - Included provided helium void fraction and ideal gas Rosenbluth weight

4. Key Simulation Parameters:
   - Cutoff distances: 12.8 Å (VDW and Coulomb)
   - Ewald precision: 1e-6
   - MC moves: Translation (50%) and Reinsertion (50%)

Files Generated:
- simulation.input: Main RASPA input file
- framework.cif: IRMOF-13 structure
- pentane.def: n-pentane molecule definition
- force_field.def: Force field parameters
- pseudo_atoms.def: Pseudoatom definitions
- force_field_mixing_rules.def: Mixing rules

To Execute:
Run 'simulate simulation.input' in RASPA to start the simulation.

Expected Output:
The simulation will calculate the total energy <U_hg> which is used to determine the adsorption enthalpy using:
ΔH = <U_hg> - RT

Note: This setup is ready for execution but has not been run yet as requested.