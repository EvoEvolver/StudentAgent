# Ideal Rosenbluth Weight Calculation for n-hexane

## Overview
This simulation calculated the ideal gas Rosenbluth weight for n-hexane using Widom insertion Monte Carlo method in RASPA.

## Why Rosenbluth Weights are Needed
- n-hexane is a flexible molecule with internal torsional degrees of freedom
- Rosenbluth weights are essential for Configurational Bias Monte Carlo (CBMC) simulations
- Required for accurate Henry coefficient calculations and GCMC adsorption simulations
- Default value of 1.0 only applies to rigid molecules without torsions

## Simulation Method
1. **Simulation Type**: Monte Carlo with Widom insertions only
2. **Key Parameters**:
   - WidomProbability: 1.0 (only virtual insertions)
   - TranslationProbability: 0.0 (disabled)
   - RotationProbability: 0.0 (disabled) 
   - SwapProbability: 0.0 (disabled)
   - CreateNumberOfMolecules: 0 (no actual molecules)
3. **Conditions**: 
   - Temperature: 298 K
   - Box size: 30×30×30 Å
   - 100,000 cycles with 10,000 initialization cycles

## Results
From the Widom insertion simulation:
- **Average Widom Rosenbluth-weight**: ~0.00245 (2.45 × 10^-3)
- This value represents the ideal gas Rosenbluth weight for n-hexane
- The low value reflects the molecular flexibility and conformational complexity

## Usage in Future Simulations
To use this result in GCMC or other CBMC simulations, add to the component definition:

```
Component 0 MoleculeName       n-hexane
            MoleculeDefinition local
            IdealGasRosenbluthWeight 0.00245
            ...
```

## Files Generated
- `n-hexane.def`: Molecule definition file
- `force_field.def`: Force field parameters
- `pseudo_atoms.def`: Pseudoatom definitions
- `simulation.input`: Widom insertion simulation input
- `output_Box_1.1.1_298.000000_100000.data`: Detailed simulation results

## Key Insights
1. Flexible molecules like n-hexane require calculated Rosenbluth weights (≠ 1.0)
2. Widom insertions provide the necessary statistical sampling without actual insertions
3. This prerequisite calculation is essential before running main adsorption simulations
4. The low Rosenbluth weight (~0.00245) indicates significant conformational restrictions

## Next Steps
Use the calculated IdealGasRosenbluthWeight (0.00245) in subsequent GCMC simulations for accurate n-hexane adsorption calculations.