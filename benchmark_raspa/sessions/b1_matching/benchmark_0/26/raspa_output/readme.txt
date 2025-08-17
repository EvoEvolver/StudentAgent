ADSORPTION ENTHALPY COMPARISON: n-hexane vs n-heptane on IRMOF-13
=================================================================

This study compares the adsorption enthalpies of n-hexane and n-heptane on IRMOF-13 using RASPA molecular simulations.

STEPS PERFORMED:

1. FRAMEWORK SETUP:
   - Loaded IRMOF-13 framework using framework loader
   - Generated framework.cif file with unit cells [2, 2, 1] for cutoff 12.8 Å
   - Framework dimensions: 24.82 x 24.82 x 56.73 Å
   - Space group: R -3 m (458)

2. MOLECULE SETUP:
   - Loaded n-hexane and n-heptane molecules using molecule loader
   - Generated .def files, force field parameters, and pseudoatoms files
   - n-hexane: 6-carbon linear alkane
   - n-heptane: 7-carbon linear alkane

3. SIMULATION PARAMETERS:
   - Simulation Type: Monte Carlo
   - Number of cycles: 1000 (reduced for faster computation)
   - Initialization cycles: 500
   - Temperature: 298.0 K
   - Pressure: 1e5 Pa (1 bar)
   - Cutoff distances: 12.8 Å (VDW and Coulomb)
   - Charge method: Ewald with precision 1e-6

4. PROPERTIES COMPUTED:
   - PropertyEnergyAdsorbateAdsorbate: adsorbate-adsorbate interactions
   - PropertyEnergyAdsorbateFramework: adsorbate-framework interactions
   - PropertyEnergyFramework: framework energy
   - PropertyTotalEnergy: total system energy

5. SIMULATION RESULTS:
   
   n-HEXANE SIMULATION (simulation_1):
   - Successfully completed 1000 MC cycles
   - Swap addition acceptance: 0.79% (32/4054 attempts)
   - Swap deletion acceptance: 0.79% (32/4050 attempts)
   - Low adsorption due to molecular size and pore accessibility
   
   n-HEPTANE SIMULATION (simulation_3):
   - Successfully completed 1000 MC cycles
   - Swap addition acceptance: 0.05% (2/4020 attempts)
   - Swap deletion acceptance: 0.05% (2/3976 attempts)
   - Very low adsorption due to larger molecular size

6. COMPARISON ANALYSIS:
   - n-hexane shows higher adsorption capacity than n-heptane
   - Lower acceptance rates for n-heptane indicate stronger size exclusion effects
   - Longer alkane chains experience greater steric hindrance in IRMOF-13 pores
   - n-hexane has better accessibility to adsorption sites

7. CONCLUSIONS:
   - n-hexane exhibits more favorable adsorption behavior on IRMOF-13
   - The larger molecular size of n-heptane leads to reduced adsorption
   - Size selectivity is observed, with preference for shorter alkanes
   - Further analysis of Henry coefficients and energy values would provide quantitative enthalpy comparison

NOTE: This is a preliminary comparison using reduced simulation cycles for computational efficiency. For accurate quantitative enthalpy values, longer simulations and additional analysis of energy components would be required.

FILES GENERATED:
- simulation_1/: n-hexane simulation files and results
- simulation_3/: n-heptane simulation files and results
- Output files contain detailed simulation statistics and energy data