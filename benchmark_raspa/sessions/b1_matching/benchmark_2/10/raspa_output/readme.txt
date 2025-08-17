# Ideal Rosenbluth Weight Calculation for n-hexane

## Purpose
This simulation calculates the ideal Rosenbluth weights for n-hexane, which are critical parameters needed for subsequent GCMC (Grand Canonical Monte Carlo) adsorption simulations.

## Steps Completed

1. **Molecule Loading**
   - Loaded n-hexane molecule definition using the Molecule loader tool
   - Generated required files: n-hexane.def, force_field.def, force_field_mixing_rules.def, pseudo_atoms.def

2. **Simulation Input File Creation**
   - Created simulation.input with the following key parameters:
     - SimulationType: MonteCarlo
     - Empty box: 30×30×30 Angström (no framework)
     - Temperature: 298 K
     - Pressure: 1×10⁵ Pa
     - Cycles: 100,000 (with 10,000 initialization)
     - WidomProbability: 1.0 (enables Widom insertion moves)
     - CreateNumberOfMolecules: 0 (no actual molecules, only probe insertions)

## What This Simulation Does
- Performs Widom insertion moves to calculate configurational accessibility
- Uses empty box to isolate molecular interactions from framework effects
- Computes energy at random insertion positions without inserting actual molecules
- Temperature-dependent calculation (must match your main simulation temperature)

## Next Steps
1. Execute the simulation using RASPA
2. Look for "Average Widom Rosenbluth-weight" in the output
3. Use this value as IdealGasRosenbluthWeight parameter in subsequent GCMC simulations

## Expected Result
For n-hexane (6-carbon chain), expect a value significantly less than 1 due to molecular complexity.

## Files Generated
- simulation.input (main input file)
- n-hexane.def (molecule definition)
- force_field.def (force field parameters)
- force_field_mixing_rules.def (mixing rules)
- pseudo_atoms.def (pseudoatom definitions)

Note: This is a prerequisite simulation that MUST be completed before running any GCMC adsorption simulations with n-hexane.