RASPA Simulation Setup for CO2 Adsorption Enthalpy on IRMOF-13
================================================================

Objective: Determine the adsorption enthalpy of CO2 on IRMOF-13
Given: Helium void fraction = 0.877

Simulation Setup (2-Step Process):

STEP 1: IdealGasRosenbluthWeight Calculation
--------------------------------------------
File: step1_rosenbluth.input
Purpose: Calculate the ideal gas Rosenbluth weight for CO2 at 298K
Method: Monte Carlo with Widom insertion (WidomProbability 1.0)
Key Parameters:
- Temperature: 298K
- No actual molecules created (CreateNumberOfMolecules 0)
- Framework: IRMOF-13 with unit cells [2,2,1]
- Cycles: 5000 (reduced from typical 50000 for speed)

STEP 2: GCMC Adsorption Simulation
----------------------------------
File: step2_gcmc.input
Purpose: Calculate CO2 adsorption isotherm and enthalpy
Method: Grand Canonical Monte Carlo (GCMC)
Key Parameters:
- Temperature: 298K
- Pressure range: 1e4 to 1e6 Pa (5 pressure points)
- SwapProbability: 1.0 (essential for GCMC)
- IdealGasRosenbluthWeight: 1.0 (placeholder - use result from Step 1)
- Framework: IRMOF-13 with helium void fraction 0.877
- Cycles: 5000 (reduced from typical 50000 for speed)

Execution Instructions:
1. Run step1_rosenbluth.input first
2. Extract IdealGasRosenbluthWeight from output
3. Update step2_gcmc.input with the calculated value
4. Run step2_gcmc.input
5. Adsorption enthalpy will be calculated automatically from energy differences

Files Created:
- framework.cif: IRMOF-13 structure
- step1_rosenbluth.input: Rosenbluth weight calculation
- step2_gcmc.input: Main GCMC adsorption simulation
- simulation.input: Current active simulation file

Note: Simulation parameters reduced to 1/10 of typical values for faster execution.
