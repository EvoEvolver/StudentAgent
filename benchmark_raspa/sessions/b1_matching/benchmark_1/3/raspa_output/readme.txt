# Ideal Rosenbluth Weight Calculation for n-hexane

## Purpose:
This simulation calculates the ideal gas Rosenbluth weights for n-hexane, which are essential prerequisite parameters for CBMC (Configurational Bias Monte Carlo) simulations.

## Steps Performed:

1. **Molecule Loading**: Loaded n-hexane molecule definition and force field files

2. **Simulation Setup**: Created simulation.input file with:
   - MonteCarlo simulation type
   - 2000 cycles (reduced from typical 20000 for faster execution)
   - 1000 initialization cycles
   - Empty box (30×30×30 Å)
   - Temperature: 298 K
   - Widom insertions (WidomProbability = 1.0)
   - No actual molecule creation (CreateNumberOfMolecules = 0)

3. **Method**: Uses Widom particle insertion method to sample configurational space without actually inserting molecules

## Expected Output:
The simulation will produce an "Average Widom Rosenbluth factor" value that represents the ideal gas Rosenbluth weight for n-hexane at 298 K.

## Usage of Results:
The calculated Rosenbluth weight should be added as "IdealGasRosenbluthWeight" parameter in future CBMC simulations involving n-hexane.

## Note:
Simulation is ready to run but not executed as requested. Use 'execute raspa' to start the calculation.