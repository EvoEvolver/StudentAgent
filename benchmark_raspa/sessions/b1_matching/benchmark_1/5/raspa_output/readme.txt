# RASPA Simulation Setup: CO2 Adsorption Enthalpy on IRMOF-13

## Objective
Determine the adsorption enthalpy of CO2 on IRMOF-13 using Monte Carlo simulation at infinite dilution conditions.

## Given Parameters
- Framework: IRMOF-13
- Molecule: CO2
- Helium void fraction: 0.877
- Simulation conditions: Infinite dilution

## Theoretical Background
Adsorption enthalpy at infinite dilution is calculated using:
ΔH = ⟨U_hg⟩ - ⟨U_h⟩ - ⟨U_g⟩ - RT

For rigid frameworks and simple molecules:
ΔH = (Total_energy - T) × R_gas_constant

## Simulation Setup Steps

### 1. Framework Loading
- Loaded IRMOF-13 framework using framework loader
- Generated framework.cif file
- Unit cells: [2, 2, 1] (sufficient for 12.8 Å cutoff)

### 2. Molecule Definition
- Attempted to load CO2 using molecule loader (failed)
- Proceeded with manual CO2 specification in input file
- Used local molecule definition approach

### 3. Input File Configuration
- **Simulation Type**: MonteCarlo (required for infinite dilution)
- **Cycles**: 1000 (reduced from typical 10000+ for faster execution)
- **Initialization**: 500 cycles
- **Temperature**: 300.0 K
- **Pressure**: 0.0 Pa (infinite dilution condition)
- **Single molecule**: CreateNumberOfMolecules 1
- **Monte Carlo moves**: Translation (50%) and Reinsertion (50%)
- **Energy computation**: Enabled for enthalpy calculation

### 4. Key Simulation Parameters
- Forcefield: local
- Charge method: Ewald with 1e-6 precision
- Cutoffs: 12.8 Å for both VDW and Coulomb
- Helium void fraction: 0.877 (given)

## Files Generated
1. `framework.cif` - IRMOF-13 structure
2. `simulation.input` - Complete simulation input file
3. `readme.txt` - This documentation

## Next Steps (NOT EXECUTED)
1. Run RASPA simulation using the generated input file
2. Extract total energy from output
3. Calculate adsorption enthalpy using: ΔH = (Total_energy - 300) × 8.314/1000 kJ/mol

## Notes
- Simulation parameters reduced for faster execution as requested
- Framework assumed rigid (typical for MOFs)
- CO2 molecule definition relies on RASPA's internal library
- Results will provide fundamental thermodynamic binding strength