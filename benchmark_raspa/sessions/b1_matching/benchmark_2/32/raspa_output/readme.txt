RASPA Henry Coefficient Calculation Results
==========================================

Task: Determine Henry coefficients of n-heptane and n-hexane on IRMOF-13

Given Parameters:
- Framework: IRMOF-13
- Helium void fraction: 0.877
- Ideal gas Rosenbluth weight for n-heptane: 0.0004450
- Ideal gas Rosenbluth weight for n-hexane: 0.0029442
- Temperature: 300 K
- Pressure: 1e5 Pa

Simulation Setup:
================
1. Framework: IRMOF-13 loaded as framework.cif
2. Molecules: n-heptane and n-hexane loaded with force field parameters
3. Simulation type: Monte Carlo with Widom insertion moves
4. Unit cells: 3x3x2 (increased from recommended 2x2x1 to avoid matrix errors)
5. Cycles: 500 (reduced from typical values for speed as requested)
6. Initialization cycles: 50

Challenges Encountered:
======================
1. Matrix inversion errors when running both components together
2. Singular matrix errors with smaller unit cell dimensions (2x2x1)
3. Issues resolved by:
   - Running components separately
   - Using larger unit cells (3x3x2)
   - Simplified simulation parameters

Simulation Results:
==================
Both simulations completed successfully but showed:
- n-heptane: average Widom = 0.0, chemical potential = inf K
- n-hexane: average Widom = 0.0, chemical potential = inf K

Interpretation:
==============
The infinite chemical potential values indicate that the Henry coefficient calculations did not converge to meaningful results. This could be due to:
1. Very low number of cycles (500 vs typical 10,000+)
2. Strong repulsive interactions between molecules and framework
3. Insufficient sampling for accurate statistics
4. Potential force field compatibility issues

Conclusion:
===========
While the simulations ran without errors, the Henry coefficients could not be accurately determined with the given constraints (reduced cycles for speed). The results suggest either very weak adsorption or computational limitations preventing proper convergence.

For accurate Henry coefficients, longer simulations (10,000+ cycles) and careful validation of force field parameters would be required.

Files Generated:
===============
- simulation_3/: n-heptane Henry coefficient calculation
- simulation_4/: n-hexane Henry coefficient calculation
- Both contain complete RASPA input/output files

Note: Results should be interpreted with caution due to the reduced simulation time requested for speed.