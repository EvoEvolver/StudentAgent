# Ideal Rosenbluth Weight Calculation for n-heptane

## Task
Calculate the ideal Rosenbluth weights for n-heptane using RASPA Monte Carlo simulation.

## Method
- Simulation Type: Monte Carlo with Widom insertion
- Molecule: n-heptane (C7H16)
- Temperature: 298 K
- Pressure: 1e5 Pa (1 bar)
- Cycles: 1000 (reduced for faster execution)
- Initialization Cycles: 100

## Steps Performed
1. Generated n-heptane molecule definition files using molecule loader
2. Created simulation input file with Widom insertion parameters
3. Executed RASPA simulation to calculate ideal gas Rosenbluth weights
4. Parsed output to extract results

## Key Parameters
- WidomProbability: 1.0 (pure Widom insertion moves)
- CreateNumberOfMolecules: 0 (no actual molecules inserted)
- Box size: 30.0 x 30.0 x 30.0 Å
- Force field: Local (generated from molecule loader)

## Results
The ideal Rosenbluth weight is calculated from the Widom insertion energy measurements and represents the reference state for subsequent CBMC simulations of n-heptane.

## Usage
This calculated ideal Rosenbluth weight can be used as input parameter (IdealGasRosenbluthWeight) for Henry coefficient calculations and GCMC adsorption simulations involving n-heptane.
