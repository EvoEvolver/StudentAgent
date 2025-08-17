# RASPA Simulation Setup: Adsorption Enthalpy Comparison of Methane and N2 on IRMOF-13

## Overview
This setup compares the adsorption enthalpies of methane and nitrogen (N2) on IRMOF-13 framework using RASPA molecular simulations.

## Prerequisites Provided
- IRMOF-13 framework (framework.cif)
- Helium void fraction: 0.877
- Methane molecule definition (methane.def)
- Nitrogen molecule definition (nitrogen.def)
- Force field parameters (force_field.def, pseudo_atoms.def, force_field_mixing_rules.def)

## Simulation Workflow

### Step 1: Calculate IdealGasRosenbluthWeight (REQUIRED FIRST)
**File:** simulation.input (in simulation_1 directory)
**Purpose:** Calculate molecular complexity correction factors for methane and nitrogen
**Method:** Widom insertions in a box
**Key Parameters:**
- SimulationType: MonteCarlo
- Widom insertions only (CreateNumberOfMolecules: 0)
- WidomProbability: 1.0 for both components
- Temperature: 298 K
- Box size: 30x30x30 Å

**Action Required:** 
1. Run this simulation first
2. Extract IdealGasRosenbluthWeight values from output for both methane and nitrogen
3. Update these values in the main GCMC simulation

### Step 2: Main GCMC Simulation for Adsorption Enthalpy
**File:** simulation_2_gcmc.input
**Purpose:** Calculate adsorption isotherms and enthalpies for both molecules
**Method:** Grand Canonical Monte Carlo
**Key Parameters:**
- Framework: IRMOF-13 with unit cells [2,2,1]
- HeliumVoidFraction: 0.877 (provided)
- Temperature: 298 K
- Pressure range: 1e3 to 1e6 Pa (7 pressure points)
- SwapProbability: 1.0 for particle insertion/deletion
- Energy and molecule histograms enabled for enthalpy calculations

**IMPORTANT:** Update IdealGasRosenbluthWeight values from Step 1 results before running!

## Expected Outputs
- Adsorption isotherms for methane and nitrogen
- Enthalpy of adsorption values for comparison
- Energy histograms for detailed analysis
- Statistical data on swap move acceptance rates

## Execution Order
1. Run simulation.input (Step 1) to get IdealGasRosenbluthWeight
2. Update simulation_2_gcmc.input with calculated values
3. Run simulation_2_gcmc.input (Step 2) for main results
4. Compare adsorption enthalpies between methane and N2

## Notes
- Unit cell dimensions ensure proper cutoff distance (>24 Å for 12.8 Å cutoff)
- Pressure range covers low to moderate pressures for accurate enthalpy determination
- Both simulations use local force field and molecule definitions
- Results will show which molecule has stronger interaction with IRMOF-13 framework
