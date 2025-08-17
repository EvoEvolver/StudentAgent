# CO2 Adsorption Enthalpy Calculation on IRMOF-13

## Task Overview
Determine the adsorption enthalpy of CO2 on IRMOF-13 given the helium void fraction of 0.877.

## Methodology
Used the infinite dilution method with a single CO2 molecule insertion in RASPA.

## Steps Performed:

1. **Framework Loading**: Loaded IRMOF-13 structure with unit cells [2,2,1] for 12.8 Å cutoff

2. **Molecule Definition**: Created CO2.def file with TraPPE force field parameters:
   - Linear CO2 molecule with C-O bond length 1.16 Å
   - Charges: C_co2 = +0.70e, O_co2 = -0.35e each
   - LJ parameters: C_co2 (ε/k_B = 27.0 K, σ = 2.80 Å), O_co2 (ε/k_B = 79.0 K, σ = 3.05 Å)

3. **Force Field Setup**: Created local force field and pseudoatoms definition files

4. **Simulation Parameters**:
   - SimulationType: MonteCarlo
   - NumberOfCycles: 50 (reduced from typical 5000 for speed)
   - NumberOfInitializationCycles: 10 (reduced from typical 1000)
   - Temperature: 298.0 K
   - Pressure: 0.0 Pa (infinite dilution)
   - HeliumVoidFraction: 0.877 (given)
   - CreateNumberOfMolecules: 1 (single molecule method)

5. **Alternative Approach Attempted**: Widom insertion method with:
   - CreateNumberOfMolecules: 0
   - WidomProbability: 1.0
   - ComputeHenryCoefficients: yes

## Issues Encountered:
- All simulation attempts were terminated ("Killed: 9") due to computational constraints
- Multiple attempts with reduced cycle numbers still failed
- Both infinite dilution and Widom insertion methods were unsuccessful

## Theoretical Calculation:
If the simulation had succeeded, adsorption enthalpy would be calculated as:

ΔH_ads = ⟨U_host-guest⟩ - ⟨U_host⟩ - ⟨U_guest⟩ - RT

Where:
- ⟨U_host-guest⟩ = average total energy of system with CO2
- ⟨U_host⟩ = average energy of empty framework
- ⟨U_guest⟩ = average energy of free CO2 molecule
- R = gas constant (8.314 J/mol·K)
- T = temperature (298.0 K)

## Expected Results:
Typical CO2 adsorption enthalpies on MOFs range from -15 to -40 kJ/mol, with IRMOF-13 expected to show moderate binding strength due to its pore structure.

## Files Created:
- framework.cif (IRMOF-13 structure)
- CO2.def (CO2 molecule definition)
- force_field.def (TraPPE parameters)
- pseudo_atoms.def (atom type definitions)
- simulation.input (RASPA input file)

## Recommendations:
- Use more computational resources or cluster computing
- Consider simplified force fields or smaller unit cells
- Try temperature-dependent isotherm measurements for enthalpy determination
- Use experimental correlations as validation

## Status: INCOMPLETE
Simulation could not be completed due to computational limitations.