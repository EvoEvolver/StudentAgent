# RASPA Simulation Setup: Adsorption Enthalpy of n-Pentane on IRMOF-13

## Objective
Determine the adsorption enthalpy of n-pentane on IRMOF-13 at infinite dilution conditions.

## Setup Steps Completed

### 1. Framework Setup
- Loaded IRMOF-13 framework using framework loader
- Framework file: framework.cif
- Unit cells: 2x2x2 (ensures >25.6Å in each dimension, more than twice the 12.8Å cutoff)
- Helium void fraction: 0.877 (provided parameter)

### 2. Molecule Setup
- Loaded n-pentane molecule definition using molecule loader
- Generated n-pentane.def file with force field parameters
- Ideal gas Rosenbluth weight: 0.0197439 (provided parameter)

### 3. Simulation Configuration
- **Simulation Type**: MonteCarlo (required for adsorption enthalpy)
- **Temperature**: 300 K
- **Pressure**: 0.0 Pa (infinite dilution conditions)
- **Cycles**: 100,000 production + 10,000 initialization
- **Initial Molecules**: 1 (single molecule insertion method)

### 4. Key Parameters for Adsorption Enthalpy
- **HeliumVoidFraction**: 0.877 (essential prerequisite)
- **IdealGasRosenbluthWeight**: 0.0197439 (critical for CBMC algorithm)
- **CreateNumberOfMolecules**: 1 (different from Henry coefficient which uses 0)
- **SwapProbability**: 0.0 (no molecule exchange at infinite dilution)

### 5. Monte Carlo Moves
- Translation: enabled (probability 1.0)
- Reinsertion: enabled (probability 1.0) 
- Partial Reinsertion: enabled (probability 1.0)
- Swap: disabled (probability 0.0)

### 6. Output Analysis
The simulation will output the total energy <U_hg> which is used to calculate:
**ΔH_ads = <U_hg> - RT**

For rigid framework and simple molecules:
- <U_h> = 0 (rigid framework)
- <U_g> = 0 (simple molecule)
- RT term accounts for ideal gas enthalpy

## Files Generated
- simulation.input: Main RASPA input file
- framework.cif: IRMOF-13 crystal structure
- n-pentane.def: Molecule definition file
- force_field_mixing_rules.def: Force field parameters
- pseudo_atoms.def: Pseudoatom definitions

## Next Steps
To execute the simulation, run: `execute raspa`
The adsorption enthalpy will be calculated from the total energy output.

## Important Notes
- Framework is treated as rigid (standard for MOF simulations)
- Infinite dilution achieved by setting external pressure to 0.0
- Single molecule insertion method provides direct thermodynamic insight
- Energy histogram computation enabled for detailed analysis
