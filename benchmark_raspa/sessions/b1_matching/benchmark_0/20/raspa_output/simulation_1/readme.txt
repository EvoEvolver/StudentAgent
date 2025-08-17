# Henry Coefficient Calculation for n-Pentane and n-Hexane on IRMOF-13

## Overview
This simulation determines the Henry coefficients of n-pentane and n-hexane adsorbed on IRMOF-13 framework using RASPA Monte Carlo simulations.

## Steps Performed

### 1. Framework Setup
- Loaded IRMOF-13 framework using framework loader tool
- Generated framework.cif file with proper crystallographic structure
- Unit cells set to [2, 2, 1] for 12.8 Angstrom cutoff compatibility

### 2. Molecule Preparation
- Generated molecular definition files for both target molecules:
  - pentane.def (n-pentane molecular geometry and properties)
  - n-hexane.def (n-hexane molecular geometry and properties)
- Created corresponding force field files:
  - force_field.def (intermolecular interaction parameters)
  - force_field_mixing_rules.def (mixing rules for different atom types)
  - pseudo_atoms.def (pseudoatom definitions)

### 3. Simulation Configuration
- **Simulation Type**: Monte Carlo (required for Henry coefficient calculations)
- **Temperature**: 298.0 K (room temperature)
- **Pressure**: 1e5 Pa (1 bar, standard conditions)
- **Cycles**: 1000 production cycles + 500 initialization cycles (reduced for faster computation)
- **Framework**: IRMOF-13 with helium void fraction of 0.8

### 4. Henry Coefficient Setup
- Configured infinite dilution conditions for Henry coefficient calculations
- Set up both components (n-pentane and n-hexane) with proper Monte Carlo moves:
  - Translation, rotation, reinsertion, and swap probabilities
  - Initial molecule count set to 0 (infinite dilution)

### 5. Key Simulation Parameters
- **Cutoff distances**: 12.8 Angstrom for both VDW and Coulomb interactions
- **Charge method**: Ewald summation with 1e-6 precision
- **Force field**: Local force field files
- **Output frequency**: Every 100 cycles

## Files Generated
- simulation.input: Main RASPA input file
- framework.cif: IRMOF-13 crystal structure
- pentane.def, n-hexane.def: Molecule definitions
- force_field.def: Force field parameters
- pseudo_atoms.def: Pseudoatom definitions
- force_field_mixing_rules.def: Mixing rules

## Expected Output
The simulation will calculate Henry coefficients for both n-pentane and n-hexane at infinite dilution conditions, providing insights into their adsorption behavior on IRMOF-13.

## Notes
- Simulation cycles reduced to 1/10 of typical values for faster execution
- All prerequisite files (framework, molecules, force fields) generated automatically
- Simulation configured for standard temperature and pressure conditions
