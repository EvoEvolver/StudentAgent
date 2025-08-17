# RASPA Simulation Setup: Adsorption Enthalpy Comparison
# n-pentane vs N2 on IRMOF-13

## Overview
This setup compares the adsorption enthalpies of n-pentane and nitrogen (N2) on the IRMOF-13 metal-organic framework using RASPA Monte Carlo simulations.

## Files Created
1. **framework.cif** - IRMOF-13 structure file
2. **pentane.def** - n-pentane molecule definition
3. **nitrogen.def** - N2 molecule definition
4. **force_field.def** - Force field parameters
5. **pseudo_atoms.def** - Pseudoatom definitions
6. **force_field_mixing_rules.def** - Mixing rules
7. **simulation_n-pentane.input** - Input file for n-pentane simulation
8. **simulation_N2.input** - Input file for N2 simulation

## Simulation Method
- **Type**: Monte Carlo with single molecule insertion
- **Purpose**: Calculate adsorption enthalpies at infinite dilution
- **Conditions**: 298 K, 0.0 Pa (infinite dilution)
- **Framework**: IRMOF-13 (rigid, HeliumVoidFraction = 0.877)

## Key Parameters Used
- **Cycles**: 5000 total (1000 initialization + 4000 production)
- **Cutoffs**: 12.8 Å for both VDW and Coulomb
- **Unit Cells**: 2×2×1 (sufficient for 12.8 Å cutoff)
- **IdealGasRosenbluthWeight**: 0.0197439 (for n-pentane only)

## How to Run Simulations

### Step 1: Run n-pentane simulation
```bash
cp simulation_n-pentane.input simulation.input
raspa simulation.input
```

### Step 2: Run N2 simulation
```bash
cp simulation_N2.input simulation.input
raspa simulation.input
```

## Results Analysis
After both simulations complete:

1. **Extract Total Energy** from each simulation output
2. **Calculate adsorption enthalpy** using:
   ΔH = ⟨U_hg⟩ - ⟨U_h⟩ - ⟨U_g⟩ - RT
   
   For rigid framework and simple molecules:
   ΔH ≈ (Total_energy - T) × R_gas_constant

3. **Compare values**: More negative ΔH indicates stronger adsorption

## Prerequisites Met
✓ Framework file (IRMOF-13) loaded
✓ Molecule definitions (n-pentane, N2) generated
✓ HeliumVoidFraction provided (0.877)
✓ IdealGasRosenbluthWeight provided for n-pentane (0.0197439)
✓ Proper Monte Carlo moves configured
✓ Infinite dilution conditions set (pressure = 0.0)

## Expected Outcome
The simulations will provide the host-guest interaction energies needed to calculate and compare the adsorption enthalpies of n-pentane and N2 on IRMOF-13, allowing determination of which molecule has stronger binding affinity to the framework.
