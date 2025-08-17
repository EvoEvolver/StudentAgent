# IRMOF-13 Helium Void Fraction Calculation Setup

## Overview
This setup calculates the helium void fraction of IRMOF-13 framework using RASPA Monte Carlo simulation with Widom particle insertion method.

## Files Created
1. **framework.cif** - IRMOF-13 framework structure
2. **helium.def** - Helium molecule definition
3. **force_field.def** - Force field parameters
4. **force_field_mixing_rules.def** - Mixing rules for interactions
5. **pseudo_atoms.def** - Pseudoatom definitions
6. **simulation.input** - Main simulation input file

## Setup Steps Completed

### Step 1: Framework Loading
- Loaded IRMOF-13 framework using framework loader
- Generated framework.cif file
- Determined required unit cells: [2, 2, 1] for cutoff 12.8 Å

### Step 2: Molecule Loading
- Loaded helium molecule definition
- Generated helium.def and associated force field files

### Step 3: Simulation Input Configuration
- **Simulation Type**: Monte Carlo
- **Cycles**: 25,000 production + 5,000 initialization
- **Method**: Widom particle insertion (WidomProbability 1.0)
- **Component**: Helium with CreateNumberOfMolecules 0
- **Framework**: IRMOF-13 with unit cells [2, 2, 1]
- **Temperature**: 298.0 K
- **Cutoffs**: 12.8 Å for both VDW and Coulomb

## Key Parameters
- **WidomProbability 1.0**: Enables Widom insertion method
- **CreateNumberOfMolecules 0**: No actual molecules inserted, only test insertions
- **UnitCells 2 2 1**: Minimum required for 12.8 Å cutoff

## Expected Output
The helium void fraction will be found as the "Average Widom Rosenbluth factor" in the simulation output file.

## Prerequisites Met
- Framework structure loaded and validated
- Helium molecule definition created
- Force field parameters established
- Simulation input file configured with correct parameters

## Next Steps
To execute the simulation, run: `execute raspa`
The helium void fraction value from this simulation will be required for any subsequent framework-based simulations with IRMOF-13.

## Important Notes
- This is a prerequisite simulation for other IRMOF-13 studies
- The helium void fraction represents accessible void space (range 0-1)
- This value is essential for excess-adsorption calculations in main simulations
