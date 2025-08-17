RASPA SIMULATION SETUP: Henry Coefficient of CO2 on IRMOF-13
=============================================================

OBJECTIVE:
Determine the Henry coefficient of CO2 on IRMOF-13 framework with helium void fraction of 0.877.

IMPORTANT: This is a TWO-STEP process that must be executed in order!

STEP 1: Calculate IdealGasRosenbluthWeight
------------------------------------------
File: step1_rosenbluth.input (or simulation.input)

Purpose: Calculate the ideal gas Rosenbluth weight for CO2 at 298K
Method: Widom insertion in empty box (30x30x30 Å)
Key Parameters:
- Empty box simulation (no framework)
- WidomProbability: 1.0
- CreateNumberOfMolecules: 0
- Temperature: 298K
- 20,000 cycles for good statistics

EXECUTION:
1. Run RASPA with step1_rosenbluth.input
2. Extract 'Average Widom Rosenbluth factor' from output
3. This value is needed for Step 2!

STEP 2: Henry Coefficient Calculation
-------------------------------------
File: step2_henry.input

Purpose: Calculate Henry coefficient using framework and Rosenbluth weight from Step 1
Method: Widom insertion in IRMOF-13 framework
Key Parameters:
- Framework: IRMOF-13 with unit cells [2,2,1]
- HeliumVoidFraction: 0.877 (given)
- Temperature: 298K (same as Step 1)
- IdealGasRosenbluthWeight: [VALUE FROM STEP 1]
- 50,000 cycles for accurate Henry coefficient

BEFORE RUNNING STEP 2:
1. Complete Step 1 first
2. Replace '[TO_BE_FILLED_FROM_STEP1]' in step2_henry.input with actual Rosenbluth weight value

FILES INCLUDED:
- framework.cif: IRMOF-13 structure
- carbon dioxide.def: CO2 molecule definition
- force_field.def: Force field parameters
- pseudo_atoms.def: Pseudoatom definitions
- force_field_mixing_rules.def: Mixing rules
- step1_rosenbluth.input: IdealGasRosenbluthWeight calculation
- step2_henry.input: Henry coefficient calculation (needs editing after Step 1)

EXPECTED OUTPUT:
Henry coefficient in [mol/kg/Pa] units from Step 2 simulation output.

NOTES:
- Both simulations use same temperature (298K) - this is critical!
- Never run Step 2 without completing Step 1 first
- The IdealGasRosenbluthWeight is temperature-dependent
- Simulations use Widom insertion (no actual molecules inserted)
