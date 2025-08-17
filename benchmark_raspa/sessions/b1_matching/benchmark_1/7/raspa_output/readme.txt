RASPA Henry Coefficient Simulation Setup for CO2 on IRMOF-13
=============================================================

This setup determines the Henry coefficient of CO2 on IRMOF-13 framework.
Given: Helium void fraction = 0.877

REQUIRED TWO-STEP PROCESS:

Step 1: IdealGasRosenbluthWeight Calculation
-------------------------------------------
- Purpose: Calculate ideal gas Rosenbluth weight for CO2 at 298K
- Simulation: Empty box (30x30x30 Å) with 1 CO2 molecule
- Input file: Current simulation.input (empty box setup)
- Run this simulation first
- Extract 'Average Widom Rosenbluth-weight' from output

Step 2: Henry Coefficient Calculation
------------------------------------
- Purpose: Calculate Henry coefficient using Widom insertion
- Framework: IRMOF-13 (framework.cif loaded with UnitCells 2 2 1)
- Temperature: 298K (room temperature)
- Method: Widom insertion moves only (no actual molecules)
- Input: Requires IdealGasRosenbluthWeight from Step 1

FILES CREATED:
- framework.cif: IRMOF-13 structure (UnitCells: 2 2 1 for 12.8 Å cutoff)
- simulation.input: Currently set for Step 1 (ideal gas calculation)

SIMULATION PARAMETERS (Reduced for speed):
- NumberOfCycles: 500 (1/10 of typical 5000)
- NumberOfInitializationCycles: 100 (1/10 of typical 1000)
- Temperature: 298K
- Cutoffs: 12.8 Å (VDW and Coulomb)

TO EXECUTE:
1. Run Step 1 simulation with current simulation.input
2. Extract IdealGasRosenbluthWeight from output
3. Modify simulation.input for Step 2:
   - Remove Box section
   - Add Framework section with IRMOF-13
   - Set CreateNumberOfMolecules 0
   - Add WidomProbability 1.0
   - Add IdealGasRosenbluthWeight [value from Step 1]
4. Run Step 2 simulation
5. Extract 'Average Henry coefficient' [mol/kg/Pa] from final output

NOTE: CO2 molecule definition may need manual setup if automatic loading fails.
Framework helium void fraction (0.877) is used in the framework definition.
