RASPA Simulation Setup for Adsorption Enthalpy Comparison
==========================================================

Objective: Compare adsorption enthalpies of n-hexane and n-pentane on IRMOF-13

Files Generated:
1. framework.cif - IRMOF-13 crystal structure
2. n-hexane.def - n-hexane molecule definition
3. pentane.def - n-pentane molecule definition  
4. force_field.def - Force field parameters
5. pseudo_atoms.def - Pseudoatom definitions
6. force_field_mixing_rules.def - Mixing rules
7. simulation.input - Input file for n-hexane adsorption
8. simulation_pentane.input - Input file for n-pentane adsorption

Simulation Parameters:
- Simulation Type: Grand Canonical Monte Carlo (GCMC)
- Temperature: 298.0 K
- Pressure range: 1e4 to 1e6 Pa (0.1 to 10 bar)
- Framework: IRMOF-13 with helium void fraction 0.877
- Unit cells: 2x2x1 (suitable for 12.8 Å cutoff)
- Cycles: 500 production + 100 initialization (reduced for speed)

Execution Steps:
1. Run simulation with simulation.input for n-hexane
2. Run simulation with simulation_pentane.input for n-pentane
3. Compare adsorption enthalpies from output files
4. Analyze energy histograms and adsorption isotherms

Note: Adsorption enthalpies will be calculated using fluctuation formulas during GCMC simulation and provided with error bars in the output.
