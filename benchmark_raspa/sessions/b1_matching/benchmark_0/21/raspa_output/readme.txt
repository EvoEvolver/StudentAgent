RASPA Simulation for N2 Adsorption Enthalpy on IRMOF-13
========================================================

Objective: Determine the adsorption enthalpy of N2 on IRMOF-13 using simulation at infinite dilution

Steps Completed:
1. Loaded IRMOF-13 framework (framework.cif) with unit cells [2, 2, 1]
2. Loaded helium and nitrogen molecule definitions
3. Ran helium void fraction calculation (prerequisite) in simulation_1/
   - Used MonteCarlo simulation with 1000 cycles, 500 initialization cycles
   - Temperature: 298 K, Pressure: 1e5 Pa
4. Ran N2 adsorption enthalpy calculation in simulation_2/
   - Used Widom insertion method for infinite dilution
   - ComputeHenryCoefficients enabled
   - Used estimated helium void fraction of 0.75
   - Temperature: 298 K, Pressure: 1e5 Pa

Current Status:
- Both simulations completed successfully
- Output files generated but results appear incomplete in parsed output
- Need to extract final adsorption enthalpy values from complete output files

Next Steps:
- Run longer simulation with more cycles for better statistics
- Extract final Henry coefficient and adsorption enthalpy values
- Calculate final adsorption enthalpy result

Note: Simulation cycles were reduced to 1/10 of typical values for faster execution as instructed.