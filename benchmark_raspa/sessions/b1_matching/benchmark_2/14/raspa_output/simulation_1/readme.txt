RASPA Henry Coefficient Simulation Setup
=========================================

Objective: Determine the Henry coefficient of n-pentane on IRMOF-13

Given Parameters:
- Helium void fraction: 0.877
- Ideal gas Rosenbluth weight for n-pentane: 0.0197439
- Framework: IRMOF-13
- Temperature: 298.0 K (standard conditions)

Files Created:
==============

1. framework.cif
   - IRMOF-13 framework structure
   - Unit cells: 2 x 2 x 1 (required for 12.8 Å cutoff)

2. pentane.def
   - n-pentane molecule definition
   - Contains atomic coordinates and connectivity

3. force_field.def
   - Force field parameters for interactions
   - Local force field specification

4. force_field_mixing_rules.def
   - Mixing rules for cross-interactions

5. pseudo_atoms.def
   - Pseudo-atom definitions for force field

6. simulation.input
   - Main RASPA input file with simulation parameters

Simulation Setup Details:
========================

Simulation Type: Monte Carlo
- Uses Widom insertion method for Henry coefficient calculation
- No actual molecules inserted (CreateNumberOfMolecules = 0)
- Only energy calculations at random positions

Key Parameters:
- Cycles: 100,000 (production run)
- Initialization: 10,000 cycles
- Widom Probability: 1.0 (pure insertion moves)
- Cutoffs: 12.8 Å (VDW and Coulomb)
- Charge Method: Ewald summation

Prerequisites Satisfied:
- Ideal gas Rosenbluth weight provided (0.0197439)
- Helium void fraction provided (0.877)
- Framework properly loaded with correct unit cells

To Execute:
===========
Run RASPA with the simulation.input file to calculate the Henry coefficient.
The result will be reported in mol/kg/Pa units.

Note: This setup is complete and ready for execution. All prerequisites
have been satisfied and all necessary files have been generated.
