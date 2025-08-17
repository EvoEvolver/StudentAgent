RASPA Henry Coefficient Calculation Setup for Methane and N2 on IRMOF-13
=========================================================================

This setup determines the Henry coefficient of methane and N2 on IRMOF-13 framework.
Given: Helium void fraction = 0.877
Temperature: 298 K

FILES CREATED:
--------------
1. framework.cif - IRMOF-13 framework structure (unit cells: 2x2x1)
2. methane.def - Methane molecule definition
3. nitrogen.def - Nitrogen molecule definition  
4. force_field.def - Force field parameters
5. force_field_mixing_rules.def - Mixing rules
6. pseudo_atoms.def - Pseudoatom definitions
7. simulation.input - Prerequisite simulation (IdealGasRosenbluthWeight calculation)
8. henry_coefficient_simulation.input - Main Henry coefficient simulation (template)

SIMULATION PROCEDURE:
--------------------

STEP 1: Calculate IdealGasRosenbluthWeight (PREREQUISITE)
- Run the current simulation.input file
- This performs Widom insertions in empty box (30x30x30 Angstrom)
- Extract 'Average Widom Rosenbluth factor' values for methane and nitrogen from output
- These values are the IdealGasRosenbluthWeight needed for Step 2

STEP 2: Main Henry Coefficient Calculation
- Update henry_coefficient_simulation.input with IdealGasRosenbluthWeight values from Step 1
- Replace [TO_BE_UPDATED_FROM_PREREQUISITE_SIMULATION] with actual values
- Run the updated henry_coefficient_simulation.input
- Extract 'Average Henry coefficient' results in [mol/kg/Pa] units

IMPORTANT NOTES:
---------------
- Both simulations use the same temperature (298 K)
- Henry coefficient simulation uses IRMOF-13 framework with given void fraction
- No molecules are actually inserted (CreateNumberOfMolecules: 0)
- Uses Widom insertion method (WidomProbability: 1.0)
- Results will show Henry coefficients for both methane and nitrogen simultaneously

TO EXECUTE:
-----------
1. First run: simulation.input (prerequisite)
2. Update henry_coefficient_simulation.input with results from step 1
3. Then run: henry_coefficient_simulation.input (main calculation)
4. Parse output files for Henry coefficient results
