RASPA Simulation Setup for Henry Coefficient of Ethane on IRMOF-13

Steps performed:
1. Loaded IRMOF-13 framework using framework loader tool
2. Generated molecule definitions for helium and ethane using molecule loader
3. Created two separate simulation input files:
   - helium_void_fraction.input: Monte Carlo simulation to calculate helium void fraction
   - ethane_henry.input: Monte Carlo simulation to calculate henry coefficient of ethane

Simulation parameters:
- Temperature: 298 K
- Cycles: 5000 (reduced to 10% of typical values for faster execution)
- Initialization cycles: 1000
- Unit cells: 2x2x1 (based on framework loader recommendation)
- Cutoff: 12.8 Å

Execution order:
1. Run helium_void_fraction.input first to get the actual helium void fraction
2. Update the HeliumVoidFraction value in ethane_henry.input with the result from step 1
3. Run ethane_henry.input to get the henry coefficient

Note: The helium void fraction in ethane_henry.input is set to 0.75 as an estimate and should be updated with the actual calculated value from the helium simulation.
