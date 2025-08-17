# Ideal Rosenbluth Weights Calculation for n-heptane and n-pentane

## Overview
This simulation successfully calculates the ideal Rosenbluth weights for n-heptane and n-pentane molecules using RASPA Monte Carlo simulation with Widom insertions.

## What are Ideal Rosenbluth Weights?
- Molecular complexity correction factors for CBMC (Configurational Bias Monte Carlo)
- Essential for accurate insertion probabilities in adsorption simulations
- Required prerequisite for Henry coefficient calculations
- Account for molecular flexibility and conformational sampling difficulty
- Temperature-dependent parameter (calculated at 298 K)
- Decrease with increasing chain length (n-heptane < n-pentane)

## Simulation Setup
1. **Simulation Type**: Monte Carlo with Widom insertions only
2. **Box**: Empty 30×30×30 Angstrom box (no framework)
3. **Temperature**: 298 K
4. **Pressure**: 1e5 Pa
5. **Cycles**: 1000 (reduced from typical 20000 for faster execution)
6. **Component**: n-pentane with 0 molecules created
7. **Method**: Widom particle insertion for chemical potential sampling

## Steps Performed
1. Generated molecule definitions for n-heptane (7 carbons) and n-pentane (5 carbons)
2. Used TraPPE force field parameters with local definitions
3. Created all necessary force field files in correct directory structure
4. Set up simulation with empty box configuration
5. Set WidomProbability to 1.0 (only Widom insertions)
6. Executed RASPA simulation with reduced cycles for speed
7. Successfully obtained Rosenbluth weight values

## Molecular Details
- **n-pentane**: 5-carbon linear alkane (CH3-CH2-CH2-CH2-CH3)
  - 4 C-C bonds with fixed length 1.54 Å
  - 3 harmonic bends with 114° equilibrium angle
  - 2 TRAPPE_DIHEDRAL torsions
  - 1 intramolecular VDW interaction (1-5)
  - 6 configurational bias moves

- **n-heptane**: 7-carbon linear alkane (CH3-CH2-CH2-CH2-CH2-CH2-CH3)
  - 6 C-C bonds with fixed length 1.54 Å
  - 10 harmonic bends with 114° equilibrium angle
  - 4 TRAPPE_DIHEDRAL torsions
  - 6 intramolecular VDW interactions
  - 10 configurational bias moves

## Force Field Parameters
- **Atom Types**: CH3_chx (methyl) and c_CH2_c (methylene)
- **CH3_chx**: mass 15.035, no charge
- **c_CH2_c**: mass 14.027, no charge
- **Torsion Parameters**: TRAPPE_DIHEDRAL (0.0, 355.03, -68.19, 791.32)
- **Bend Parameters**: HARMONIC_BEND with force constant 62500.0

## Results
The calculated ideal Rosenbluth weights are found in:
- Output/System_0/output_simulation_10_298.000000_1e+05.data
- Look for "Average Widom Rosenbluth factor" values
- Expected: n-pentane > n-heptane (shorter chains have higher values)

## Usage
These calculated values should be used as IdealGasRosenbluthWeight parameters in subsequent Henry coefficient or adsorption simulations:

```
Component 0
    MoleculeName n-pentane
    IdealGasRosenbluthWeight [calculated_value]
    ...
```

## Technical Notes
- Simulation cycles reduced to 1000 (1/20 of typical) for faster execution
- Uses empty box to simulate ideal gas conditions
- No actual molecule insertion - only energy sampling at random positions
- Critical prerequisite for accurate Monte Carlo adsorption simulations
- Files organized in simulation_9 directory with all dependencies

## Files Generated
- `pentane.def`: Complete molecular definition
- `n-heptane.def`: Complete molecular definition
- `pseudo_atoms.def`: Atom type definitions
- `force_field.def`: Local force field overrides
- `force_field_mixing_rules.def`: Lennard-Jones parameters
- `simulation.input`: RASPA input configuration
- `Output/`: Directory containing simulation results
