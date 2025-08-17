RASPA Henry Coefficient Simulation Setup
=========================================

Task: Determine the Henry coefficient of n-pentane on IRMOF-13

Given Parameters:
- Helium void fraction of IRMOF-13: 0.877
- Ideal gas Rosenbluth weight for n-pentane: 0.0197439

Steps Completed:

1. Framework Loading:
   - Loaded IRMOF-13 framework as framework.cif
   - Recommended unit cells for 12.8 Å cutoff: [2, 2, 1]

2. Molecule Loading:
   - Loaded n-pentane molecule definition as pentane.def
   - Generated corresponding force field files

3. Simulation Input File Creation:
   - Simulation Type: MonteCarlo
   - Cycles: 500 (production) + 100 (initialization) - reduced for faster execution
   - Method: Widom insertion (WidomProbability 1.0)
   - No actual molecules inserted (CreateNumberOfMolecules 0)
   - Used provided IdealGasRosenbluthWeight: 0.0197439
   - Framework: IRMOF-13 with unit cells [2,2,1]
   - Helium void fraction: 0.877
   - Temperature: 298 K
   - Pressure: 1e5 Pa

Files Generated:
- simulation.input (main input file)
- framework.cif (IRMOF-13 structure)
- pentane.def (n-pentane molecule definition)
- force_field.def (force field parameters)
- pseudo_atoms.def (pseudoatom definitions)
- force_field_mixing_rules.def (mixing rules)

To Execute:
Run 'execute raspa' to start the simulation.

Expected Output:
The simulation will calculate the Henry coefficient of n-pentane in IRMOF-13 in units of [mol/kg/Pa].

Note: Cycle numbers were reduced to 1/10 of typical values for faster execution as requested.
