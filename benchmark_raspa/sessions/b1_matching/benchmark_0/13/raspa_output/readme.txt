# Adsorption Enthalpy Calculation of N2 on IRMOF-13

## Overview
This project calculates the adsorption enthalpy of nitrogen (N2) on IRMOF-13 using RASPA molecular simulation software.

## Files Generated

### Framework and Molecule Files:
- `framework.cif`: IRMOF-13 crystal structure (unit cells: 2x2x1 for 12.8 Å cutoff)
- `nitrogen.def`: N2 molecule definition file
- `helium.def`: Helium molecule definition file (for void fraction calculation)
- `force_field.def`: Force field parameters
- `force_field_mixing_rules.def`: Mixing rules for interactions
- `pseudo_atoms.def`: Pseudoatom parameters

### Simulation Input Files:
1. `helium_voidfraction.input`: Prerequisite helium void fraction calculation at 298K
2. `simulation.input`: Main N2 adsorption simulation at 298K
3. `simulation_273K.input`: N2 adsorption simulation at 273K
4. `simulation_323K.input`: N2 adsorption simulation at 323K

## Procedure

### Step 1: Prerequisites
1. **Framework Loading**: Generated IRMOF-13.cif using framework loader
2. **Molecule Loading**: Generated nitrogen.def and helium.def using molecule loader
3. **Force Field Setup**: Automatically generated force field and pseudoatom files

### Step 2: Helium Void Fraction Calculation (MANDATORY FIRST)
Run `helium_voidfraction.input` first to determine the framework's void fraction.
This is a prerequisite for accurate adsorption calculations.

### Step 3: Temperature-Dependent N2 Adsorption Simulations
Run the following simulations in sequence:
1. `simulation_273K.input` (273K)
2. `simulation.input` (298K) 
3. `simulation_323K.input` (323K)

### Step 4: Adsorption Enthalpy Calculation
Adsorption enthalpy is calculated from the temperature dependence of adsorption:
- Use the Van't Hoff equation: d(ln K)/d(1/T) = -ΔH_ads/R
- Where K is the Henry coefficient from each temperature simulation
- Plot ln(K) vs 1/T; slope gives -ΔH_ads/R

## Simulation Parameters
- **Simulation Type**: Monte Carlo
- **Cycles**: 1,000 (reduced for faster computation)
- **Initialization Cycles**: 500
- **Temperature Range**: 273K - 323K
- **Pressure**: 1×10⁵ Pa (1 bar)
- **Cutoff**: 12.8 Å (VDW and Coulomb)
- **Framework Unit Cells**: 2×2×1

## Key Properties Computed
- Energy histograms
- Number of molecules histograms
- Henry coefficients (for enthalpy calculation)
- Adsorption isotherms

## Expected Outputs
After running simulations, analyze:
1. Henry coefficients from each temperature
2. Energy distributions
3. Adsorption amounts vs temperature
4. Calculate enthalpy from temperature dependence

## Notes
- Helium void fraction must be determined first and used in N2 simulations
- Reduced simulation cycles (1,000) used for faster computation
- All simulations use local force field and framework definitions
- Results should be analyzed using RASPA output parser tools

## Execution Order
1. Run helium_voidfraction.input
2. Update HeliumVoidFraction value in N2 simulation files
3. Run N2 simulations at all three temperatures
4. Analyze results to calculate adsorption enthalpy
