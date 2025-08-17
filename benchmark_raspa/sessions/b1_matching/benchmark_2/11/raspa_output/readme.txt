# Ideal Rosenbluth Weights Calculation Setup

## Purpose
This simulation calculates the ideal Rosenbluth weights for n-hexane and pentane molecules using the Widom particle insertion method. These weights are essential prerequisites for CBMC (Configurational Bias Monte Carlo) simulations.

## Files Generated
1. **simulation.input** - Main RASPA input file for Rosenbluth weight calculation
2. **n-hexane.def** - Molecule definition for n-hexane (6 carbon atoms)
3. **pentane.def** - Molecule definition for pentane (5 carbon atoms)
4. **force_field_mixing_rules.def** - Force field mixing rules
5. **pseudo_atoms.def** - Pseudoatom definitions
6. **force_field.def** - Force field parameters

## Simulation Setup Details
- **Simulation Type**: Monte Carlo
- **Method**: Widom particle insertion (WidomProbability = 1.0)
- **Box**: Empty 30×30×30 Angstrom box (no framework)
- **Temperature**: 298 K
- **Pressure**: 1×10⁵ Pa
- **Cycles**: 20,000 production cycles + 10,000 initialization cycles
- **Molecules**: No actual insertion (CreateNumberOfMolecules = 0)

## Molecule Details
- **n-hexane**: 6-carbon alkane chain with 5 bonds, 8 bends, 3 torsions
- **pentane**: 5-carbon alkane chain with 4 bonds, 3 bends, 2 torsions
- Both use TRAPPE force field with flexible torsional degrees of freedom

## Key Parameters
- Both molecules included as separate components in same simulation
- Uses local force field definitions
- Ewald summation for electrostatics (precision 1e-6)
- 12.8 Angstrom cutoffs for VDW and Coulomb interactions
- Print results every 1000 cycles

## Expected Output
The simulation will output 'Average Widom Rosenbluth factor' values for both molecules:
- These values represent the IdealGasRosenbluthWeight parameters
- Values will be between 0 and 1 (closer to 0 for more complex molecules)
- n-hexane will have lower value than pentane due to increased complexity

## How to Execute
1. Navigate to the simulation_1 directory
2. Run: `simulate simulation.input`
3. Wait for completion (should take minutes to hours depending on system)
4. Extract Rosenbluth weights from output files

## Next Steps After Execution
1. Parse the output to extract the Rosenbluth weight values
2. Record these values for use in future CBMC simulations
3. Use these weights in component definitions for adsorption simulations

## Important Notes
- This is a **prerequisite calculation** - must be completed before main simulations
- The calculated weights are temperature-dependent (calculated at 298 K)
- Essential for accurate CBMC sampling of flexible alkane conformations
- No framework interactions - pure gas-phase molecular sampling
