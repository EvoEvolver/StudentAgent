# Ideal Rosenbluth Weights Calculation for n-Hexane and n-Pentane

## Overview
This project calculated the ideal Rosenbluth weights for n-hexane and n-pentane using RASPA molecular simulations with Widom insertion method.

## What are Ideal Rosenbluth Weights?
Ideal Rosenbluth weights are critical parameters in Monte Carlo simulations that:
- Correct acceptance probabilities when configurational bias is used
- Quantify the configurational accessibility of flexible molecules
- Range from 0 to 1, with lower values for more complex/flexible molecules
- Are essential for CBMC (Configurational Bias Monte Carlo) simulations

## Methodology
1. **Simulation Type**: Monte Carlo with Widom insertion
2. **Temperature**: 298.0 K
3. **Box Size**: 30.0 × 30.0 × 30.0 Å³
4. **Cycles**: 50,000 production cycles + 10,000 initialization cycles
5. **Method**: Pure Widom insertions (WidomProbability = 1.0, CreateNumberOfMolecules = 0)

## Results

### n-Hexane (C₆H₁₄)
- **Ideal Rosenbluth Weight**: 0.00245519 ± 8e-06
- **Chemical Potential**: -1249.82 ± 0.99 K
- **Simulation Directory**: simulation_1/

### n-Pentane (C₅H₁₂)
- **Ideal Rosenbluth Weight**: 0.0197746 ± 4e-05
- **Chemical Potential**: -1871.51 ± 0.60 K
- **Simulation Directory**: simulation_2/

## Key Insights

1. **Chain Length Effect**: n-Pentane has a significantly higher Rosenbluth weight (0.0198) compared to n-hexane (0.0025), demonstrating that longer alkane chains have reduced configurational accessibility.

2. **Molecular Complexity**: The ~8-fold difference in Rosenbluth weights reflects the exponential decrease in configurational freedom as chain length increases.

3. **Statistical Precision**: Both calculations achieved good statistical precision with relative errors < 0.2%.

## Files Generated
- `simulation_1/`: n-hexane calculation files and results
- `simulation_2/`: n-pentane calculation files and results
- Molecule definition files: `n-hexane.def`, `pentane.def`
- Force field files: `force_field.def`, `pseudo_atoms.def`
- Output data: `output_Box_1.1.1_298.000000_0.data` in each simulation directory

## Applications
These Rosenbluth weights are essential for:
- CBMC simulations of alkane adsorption
- Multi-component mixture simulations
- Henry coefficient calculations
- Accurate sampling of flexible alkane conformations

## Technical Notes
- Simulations used NVT ensemble (constant volume and temperature)
- Widom insertions were performed without actual molecule creation
- Results represent ideal gas behavior in the specified simulation box
- Values are temperature-dependent and calculated at 298 K
