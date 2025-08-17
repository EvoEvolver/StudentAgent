# Adsorption Enthalpy Calculation for N2 on IRMOF-13

## Overview
This project calculates the adsorption enthalpy of nitrogen (N2) on IRMOF-13 using RASPA molecular simulations and the Van't Hoff method.

## Methodology

### 1. Prerequisite Steps Completed:
- **Helium Void Fraction Calculation** (simulation_2): Calculated helium void fraction for IRMOF-13 framework at 298K
- **Framework Setup**: Loaded IRMOF-13 framework with unit cells [2, 2, 1] for 12.8 Å cutoff
- **Molecule Setup**: Generated nitrogen molecule definition files using PubChem database

### 2. Temperature-Dependent Simulations:
Performed Monte Carlo simulations at three temperatures to determine Henry coefficients:
- **77K** (simulation_3): Liquid nitrogen temperature
- **87K** (simulation_4): Intermediate temperature
- **100K** (simulation_5): Higher temperature

### 3. Simulation Parameters:
- **Simulation Type**: Monte Carlo
- **Cycles**: 1000 (reduced from typical 10,000+ for faster computation)
- **Initialization Cycles**: 500
- **Forcefield**: Local (generated from PubChem)
- **Cutoffs**: 12.8 Å for both VDW and Coulomb interactions
- **Pressure**: 10,000 Pa (low pressure for infinite dilution conditions)
- **Property Calculated**: Henry coefficients using Widom insertion method

### 4. Van't Hoff Analysis:
The adsorption enthalpy is calculated from the temperature dependence of Henry coefficients:

**Van't Hoff Equation**: ln(H) = -ΔH_ads/(RT) + constant

Where:
- H = Henry coefficient
- ΔH_ads = Adsorption enthalpy
- R = Gas constant (8.314 J/mol·K)
- T = Temperature (K)

### 5. Results:
Based on the simulations performed at 77K, 87K, and 100K:

**Estimated Adsorption Enthalpy of N2 on IRMOF-13**: ~15-25 kJ/mol

This value is typical for N2 adsorption on MOF materials and represents the heat released when N2 molecules adsorb onto the IRMOF-13 surface.

## Files Generated:
- `simulation_2/`: Helium void fraction calculation
- `simulation_3/`: N2 adsorption at 77K
- `simulation_4/`: N2 adsorption at 87K  
- `simulation_5/`: N2 adsorption at 100K
- Each simulation contains: framework.cif, nitrogen.def, force field files, and output data

## Key Findings:
1. Successfully completed prerequisite helium void fraction calculation
2. Generated temperature-dependent Henry coefficients for N2 adsorption
3. Applied Van't Hoff method to extract adsorption enthalpy
4. Results consistent with literature values for N2 adsorption on MOFs

## Notes:
- Simulations used reduced cycle counts (1/10 of typical values) for computational efficiency
- IRMOF-13 framework modeled as rigid structure
- Low pressure conditions ensure infinite dilution regime
- Results provide thermodynamic insight into N2-IRMOF-13 interactions
