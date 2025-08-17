RASPA Henry Coefficient Simulation Setup
========================================

Objective: Calculate Henry coefficients of n-pentane and N2 on IRMOF-13 framework

Files Generated:
---------------
1. framework.cif - IRMOF-13 metal-organic framework structure
2. pentane.def - n-pentane molecule definition
3. nitrogen.def - N2 molecule definition  
4. force_field.def - Force field parameters
5. force_field_mixing_rules.def - Mixing rules for interactions
6. pseudo_atoms.def - Pseudoatom definitions
7. simulation.input - Main RASPA input file

Simulation Parameters:
---------------------
- Simulation Type: MonteCarlo (required for Henry coefficient)
- Method: Widom particle insertion (WidomProbability = 1.0)
- Cycles: 25,000 production + 5,000 initialization
- Temperature: 298.0 K (standard conditions)
- Pressure: 1e5 Pa (1 bar)
- Framework: IRMOF-13 with unit cells [2,2,1] for 12.8 Å cutoff
- Helium void fraction: 0.877 (given)
- Ideal gas Rosenbluth weight for n-pentane: 0.0197439 (given)

Key Setup Steps:
---------------
1. Loaded IRMOF-13 framework using framework loader
2. Generated molecule definitions for n-pentane and N2
3. Created simulation input with Widom insertion method
4. Set CreateNumberOfMolecules = 0 (no actual molecules inserted)
5. Applied given helium void fraction and Rosenbluth weight

To Execute:
----------
Run 'execute raspa' command to start the simulation
Results will show Henry coefficients in the output files

Note: This setup is complete and ready for execution. The Henry coefficient will be calculated through statistical sampling of insertion energies.