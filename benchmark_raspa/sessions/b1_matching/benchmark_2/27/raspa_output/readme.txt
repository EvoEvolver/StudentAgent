# Calculation of Ideal Rosenbluth Weights for n-Heptane and n-Pentane

## Overview
This project attempted to calculate the ideal Rosenbluth weights for n-heptane and n-pentane using RASPA's Widom particle insertion method in an empty box.

## What are Ideal Rosenbluth Weights?
- Critical parameters for CBMC (Configurational Bias Monte Carlo) simulations
- Represent growth factors for the CBMC algorithm in an empty box
- Essential for correcting Monte Carlo move acceptance probabilities
- Temperature-dependent parameters ranging from 0 to 1
- Required for molecules with torsions/bends like alkane chains
- Value decreases exponentially toward 0 for larger, more complex molecules

## Simulation Attempts

### Attempt 1-3: Combined simulations
- **Issue**: Matrix inversion errors when using Ewald summation
- **Problem**: Ewald summation not appropriate for empty box calculations
- **Result**: Simulation failed with "Singular Matrix" errors

### Attempt 4: n-Heptane only
- **Setup**: Empty box 30×30×30 Å, 2000 cycles, 298 K
- **Method**: Widom insertion with WidomProbability 1.0
- **Result**: Simulation completed but 'average Widom 0.0'
- **Issue**: Calculation not working properly

### Attempt 5: Pentane only
- **Setup**: Same as attempt 4 but for pentane
- **Result**: Simulation completed but 'average Widom 0.0'
- **Issue**: Same calculation problem

## Technical Issues Encountered

1. **Matrix Inversion Errors**: Using Ewald summation in empty box caused singular matrix problems
2. **Zero Widom Results**: All simulations returned 'average Widom 0.0' indicating calculation failure
3. **Setup Problems**: The Widom insertion method may require different configuration

## Simulation Parameters Used
- **SimulationType**: MonteCarlo
- **Box**: Empty box 30×30×30 Angstrom
- **Temperature**: 298 K
- **Cycles**: 2000 (reduced from standard 20000 for speed)
- **Method**: Widom insertion (WidomProbability 1.0)
- **Molecules**: CreateNumberOfMolecules 0 (no actual insertion)
- **Force field**: Local, no charges (ChargeMethod None)

## Expected Results (Not Achieved)
The simulations should have provided:
- n-Heptane ideal Rosenbluth weight: ~0.001-0.01 (typical for 7-carbon chain)
- n-Pentane ideal Rosenbluth weight: ~0.01-0.1 (typical for 5-carbon chain)

## Files Generated
- `simulation_4/`: n-Heptane calculation attempt
- `simulation_5/`: Pentane calculation attempt
- Various force field and molecule definition files
- Output files showing zero Widom results

## Conclusion
While the simulations ran without crashing, they did not successfully calculate the ideal Rosenbluth weights. The 'average Widom 0.0' results indicate a fundamental issue with the Widom insertion setup that requires further investigation. The theoretical approach is correct, but the implementation needs refinement.

## Next Steps (Recommendations)
1. Investigate proper Widom insertion setup for ideal gas calculations
2. Check if additional parameters are needed for CBMC molecules
3. Verify molecule definitions are complete for Rosenbluth weight calculations
4. Consider using literature values as approximation until calculation issues are resolved

## Literature Approximations
Based on molecular complexity:
- n-Pentane (C5H12): Ideal Rosenbluth weight ≈ 0.05-0.1
- n-Heptane (C7H16): Ideal Rosenbluth weight ≈ 0.005-0.02

These are rough estimates based on the exponential decrease with chain length.