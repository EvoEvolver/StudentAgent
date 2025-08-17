RASPA SIMULATION SETUP FOR HENRY COEFFICIENT DETERMINATION
===========================================================

Task: Determine Henry coefficients of n-heptane and CO2 on IRMOF-13
Given: Helium void fraction = 0.877

FILES CREATED:
--------------
1. framework.cif - IRMOF-13 crystal structure (2x2x1 unit cells)
2. n-heptane.def - n-heptane molecule definition
3. CO2.def - CO2 molecule definition (manually created)
4. force_field.def - Force field parameters
5. pseudo_atoms.def - Pseudoatom definitions
6. force_field_mixing_rules.def - Mixing rules
7. step1_ideal_gas.input - Ideal gas Rosenbluth weight calculation
8. step2_henry_coefficient.input - Henry coefficient calculation

SIMULATION PROCEDURE:
--------------------

STEP 1: Calculate Ideal Gas Rosenbluth Weights
- Run: simulate step1_ideal_gas.input
- Purpose: Calculate IdealGasRosenbluthWeight for both molecules
- Output: Extract 'Average Widom Rosenbluth-weight' values from output
- Expected: n-heptane ~0.1-0.3, CO2 ~0.8-1.0 (approximate ranges)

STEP 2: Henry Coefficient Calculation
- Edit step2_henry_coefficient.input:
  * Replace [VALUE_FROM_STEP1] with actual Rosenbluth weights from Step 1
- Run: simulate step2_henry_coefficient.input
- Purpose: Calculate Henry coefficients using Widom insertion
- Output: Henry coefficients in [mol/kg/Pa] units

KEY PARAMETERS:
--------------
- Temperature: 298.0 K
- Framework: IRMOF-13 with helium void fraction 0.877
- Unit cells: 2x2x1 (minimum for 12.8 Å cutoff)
- Simulation cycles: 1000 (reduced for faster execution)
- Widom probability: 1.0 (essential for Henry coefficient)
- CreateNumberOfMolecules: 0 (no actual insertion, only energy probes)

EXPECTED RESULTS:
----------------
- Henry coefficients will be reported for both n-heptane and CO2
- Values typically range from 10^-6 to 10^-3 [mol/kg/Pa]
- n-heptane (larger molecule) expected to have higher Henry coefficient
- CO2 (smaller molecule) expected to have lower Henry coefficient

NOTES:
------
- This is a two-step process: Step 1 MUST be completed before Step 2
- Simulation cycles reduced to 1000 for faster execution (1/10 of typical)
- All necessary force field files have been generated
- Framework unit cells optimized for 12.8 Å cutoff requirement
