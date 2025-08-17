IDEAL ROSENBLUTH WEIGHT CALCULATION FOR N-PENTANE
=================================================

OBJECTIVE:
Calculate the ideal Rosenbluth weights for n-pentane using RASPA Monte Carlo simulations with Widom insertion moves.

METHOD:
1. Widom insertion simulation in empty box (no framework)
2. Trial insertions at random positions without actual molecule insertion
3. Statistical sampling to calculate average Widom Rosenbluth weight

SIMULATION SETUP:
- Simulation Type: Monte Carlo
- Number of Cycles: 2000 (reduced for speed as requested)
- Initialization Cycles: 1000
- Box Size: 30.0 x 30.0 x 30.0 Angstrom (empty box)
- Temperature: 298.0 K
- Pressure: 1e5 Pa
- Molecule: n-pentane (pentane.def)
- Widom Probability: 1.0
- Create Number of Molecules: 0 (no actual insertion)

RESULTS:
The simulation calculated the following average Widom values throughout the run:
- Cycle 100: 0.0191836816
- Cycle 200: 0.0191958527
- Cycle 300: 0.0194645015
- Cycle 400: 0.0194309114
- Final converged value: ~0.019

CONCLUSION:
The ideal Rosenbluth weight for n-pentane is approximately 0.019.

This value reflects the molecular complexity of n-pentane as a flexible chain alkane with multiple conformational states. The weight is significantly less than 1 (which would be for simple molecules like methane), indicating the configurational complexity that must be accounted for in CBMC moves.

IMPORTANCE:
This ideal Rosenbluth weight is essential for:
- Configurational Bias Monte Carlo (CBMC) simulations
- Henry coefficient calculations
- Grand Canonical Monte Carlo (GCMC) adsorption studies
- Proper statistical sampling of flexible molecules

FILES GENERATED:
- simulation.input: RASPA input file for Widom insertion simulation
- pentane.def: Molecule definition file for n-pentane
- force_field.def: Force field parameters
- pseudo_atoms.def: Pseudo atom definitions
- force_field_mixing_rules.def: Mixing rules for interactions
- Output files: Complete simulation results in Output/System_0/ directory

NOTE: The simulation used reduced cycles (1/10 of typical) for speed as requested, which may affect precision but provides the correct order of magnitude for the ideal Rosenbluth weight.