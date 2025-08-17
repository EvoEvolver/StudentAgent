RASPA Simulation Setup: Ideal Rosenbluth Weights Calculation
============================================================

Purpose:
Calculate ideal Rosenbluth weights for n-heptane and n-pentane molecules using Widom insertions.
These weights are essential correction factors for Monte Carlo move acceptance probabilities when using CBMC (Configurational Bias Monte Carlo).

Simulation Details:
- Simulation Type: Monte Carlo with Widom insertions
- Molecules: n-heptane and n-pentane
- Method: Widom insertions (WidomProbability = 1.0)
- No actual molecules created (CreateNumberOfMolecules = 0)
- Box size: 30×30×30 Å (meets >2×cutoff requirement)
- Temperature: 298 K
- Cycles: 50,000 (reduced from typical 500,000 for faster execution)
- Initialization: 100 cycles

Files Generated:
1. simulation.input - Main simulation input file
2. n-heptane.def - Molecule definition for n-heptane
3. n-pentane.def - Molecule definition for n-pentane
4. force_field_mixing_rules.def - Force field parameters
5. pseudo_atoms.def - Pseudoatom definitions

Next Steps:
1. Execute the simulation using RASPA
2. Extract the average Widom Rosenbluth weights from the output
3. Use these values as IdealGasRosenbluthWeight parameters in subsequent simulations

Note: The calculated weights will be between 0 and 1, with smaller values for more complex molecules.
