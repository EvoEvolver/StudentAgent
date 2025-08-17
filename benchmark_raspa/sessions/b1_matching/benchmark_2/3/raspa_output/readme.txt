# RASPA Simulation Setup: CO2 Adsorption Enthalpy on IRMOF-13 at Infinite Dilution

## Overview
This simulation setup determines the adsorption enthalpy of CO2 on IRMOF-13 framework at infinite dilution conditions using Monte Carlo simulation in RASPA.

## Theoretical Background
Adsorption enthalpy at infinite dilution is calculated using:
ΔH = ⟨U_hg⟩ - ⟨U_h⟩ - ⟨U_g⟩ - RT

For rigid frameworks and simple molecules, this simplifies to:
ΔH = (Total_energy - T) × R_gas_constant

Where:
- Total_energy: Average energy from simulation output
- T: Temperature (300 K)
- R_gas_constant: 8.314462618 J/(mol·K)

## Files Generated

### 1. framework.cif
- IRMOF-13 crystal structure
- Unit cells: [2, 2, 1] (minimum for 12.8 Å cutoff)
- Helium void fraction: 0.877 (given)

### 2. carbon dioxide.def
- CO2 molecule definition with force field parameters
- Contains bonded and non-bonded interaction parameters

### 3. force_field.def & force_field_mixing_rules.def
- Force field parameters for framework-molecule interactions
- Mixing rules for cross-interactions

### 4. pseudo_atoms.def
- Atomic parameters for all atom types

### 5. simulation.input
- Main simulation input file with infinite dilution parameters:
  - SimulationType: MonteCarlo
  - CreateNumberOfMolecules: 1 (single molecule)
  - ExternalPressure: 0.0 (infinite dilution)
  - Temperature: 300 K
  - 100,000 cycles with 10,000 initialization

## Key Simulation Parameters
- **Infinite Dilution**: ExternalPressure = 0.0, single molecule insertion
- **Temperature**: 300 K
- **Cutoffs**: 12.8 Å for both VDW and Coulomb interactions
- **MC Moves**: Translation (1.0) and Reinsertion (1.0) probabilities

## How to Run
1. All required files are generated in the simulation directory
2. Execute: `raspa simulation.input`
3. Monitor output for 'Total energy' values

## Results Interpretation
1. Extract 'Total energy' from simulation output (average value)
2. Calculate adsorption enthalpy:
   ΔH = (Total_energy - 300) × 8.314462618/1000 kJ/mol
3. Negative values indicate favorable adsorption

## Prerequisites Met
- ✓ IRMOF-13 framework loaded with correct unit cells
- ✓ CO2 molecule definition with force field
- ✓ Helium void fraction provided (0.877)
- ✓ Infinite dilution conditions configured
- ✓ Appropriate cutoffs and simulation parameters

## Notes
- Simulation is ready to execute but not run per instructions
- Results will provide thermodynamic insight into CO2-IRMOF-13 binding strength
- For validation, ensure energy drift < 1e-3 during simulation
