# RASPA Simulation: Adsorption Enthalpies of n-pentane/N2 Mixture on IRMOF-13

## Simulation Setup:
- Framework: IRMOF-13 (unit cells: 2x2x2)
- Temperature: 300 K
- Pressure: 1e4 Pa (10 kPa)
- Helium void fraction: 0.877
- Mixture: 1:1 n-pentane and N2 (5 molecules each)
- Simulation type: Monte Carlo
- Cycles: 2500 (reduced from typical 50000 for speed)
- Initialization cycles: 1000

## Key Parameters:
- n-pentane ideal gas Rosenbluth weight: 0.0197439
- Cutoff distances: 12.8 Å (VDW and Coulomb)
- Charge method: Ewald summation

## Files Generated:
1. framework.cif - IRMOF-13 structure
2. pentane.def - n-pentane molecule definition
3. nitrogen.def - N2 molecule definition
4. simulation.input - RASPA input file
5. Output files with adsorption enthalpy results

## Simulation Steps:
1. Loaded IRMOF-13 framework
2. Generated molecule definitions for pentane and nitrogen
3. Created Monte Carlo simulation input with enthalpy calculation enabled
4. Executed RASPA simulation
5. Parsed results for adsorption enthalpies

## Notes:
- Used reduced cycle count (1/20) for speed as requested
- Simulation includes energy histograms and enthalpy of adsorption calculations
- Results provide comparative adsorption behavior of the two components
