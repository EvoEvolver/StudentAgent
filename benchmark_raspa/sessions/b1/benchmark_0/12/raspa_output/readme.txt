RASPA Simulation Setup for Ethane Adsorption Enthalpy on IRMOF-13
==================================================================

Objective: Calculate the adsorption enthalpy of ethane on IRMOF-13 framework

Steps Completed:
1. Loaded IRMOF-13 framework (framework.cif) with unit cells [2,2,1] for 14.0 Å cutoff
2. Generated molecule definitions for ethane and helium
3. Created simulation input files

Required Simulation Workflow:

Step 1: Calculate Helium Void Fraction
- Run Monte Carlo simulation with helium to determine void fraction
- Use the current simulation.input (configured for helium void fraction calculation)
- Extract void fraction value from output

Step 2: Calculate Adsorption Enthalpy
- Modify simulation.input to use ethane instead of helium
- Set HeliumVoidFraction to value obtained from Step 1
- Run simulations at two different temperatures (298K and 318K)
- Use multiple pressure points for better statistics
- Calculate isosteric heat of adsorption from temperature dependence

Current Configuration:
- Framework: IRMOF-13 with unit cells [2,2,1]
- Cutoff: 12.8 Å for both VDW and Coulomb
- Simulation cycles: 1000 (reduced to 10% as requested)
- Initialization cycles: 500
- Molecules: ethane.def and helium.def available
- Force field: local force field files generated

Note: The simulation is currently set up for ethane adsorption at 298K. 
To complete the enthalpy calculation, you need to:
1. First run helium void fraction calculation
2. Update HeliumVoidFraction parameter with the result
3. Run ethane adsorption at both 298K and 318K
4. Calculate enthalpy from ln(P) vs 1/T plot slope

Formula for adsorption enthalpy:
ΔH_ads = -R * d(ln P)/d(1/T) at constant loading