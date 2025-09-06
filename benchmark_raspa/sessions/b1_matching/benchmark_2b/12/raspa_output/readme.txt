# Adsorption Enthalpy Calculation for Methane on IRMOF-13

## Objective
Determine the adsorption enthalpy of methane on IRMOF-13 at:
- Pressure: 1e5 Pa (100 kPa)
- Temperature: 300 K
- Given helium void fraction: 0.877

## Simulation Setup Steps

### 1. Framework Loading
- Loaded IRMOF-13 framework using framework loader
- Generated framework.cif file
- Recommended unit cells: [2, 2, 1] for cutoff 12.8 Å

### 2. Molecule Setup
- Generated methane molecule files:
  - methane.def (molecule definition)
  - force_field.def (force field parameters)
  - pseudo_atoms.def (pseudoatom definitions)
  - force_field_mixing_rules.def

### 3. Simulation Parameters
- Simulation Type: Monte Carlo (GCMC)
- Number of Cycles: 25,000
- Initialization Cycles: 5,000
- Print Every: 1,000 cycles
- Force Field: Local
- Charge Method: Ewald (precision 1e-6)
- Cutoffs: VDW and Coulomb both 12.8 Å

### 4. Monte Carlo Moves
- Translation Probability: 0.5
- Rotation Probability: 0.5
- Reinsertion Probability: 0.5
- Swap Probability: 1.0 (essential for GCMC)

### 5. Framework Specifications
- Framework Name: framework (framework.cif)
- Unit Cells: 2 2 1
- Helium Void Fraction: 0.877 (provided)
- External Temperature: 300.0 K
- External Pressure: 1e5 Pa

## Results

### Key Finding: Adsorption Enthalpy
**Enthalpy of adsorption: -17.40 ± 0.28 kJ/mol**
(or -2092.75 ± 33.24 K)

### Additional Results
- Average density: 16.31 ± 0.16 kg/m³
- Average volume: 121,087.52 ± 0.00 Ų
- Heat capacity: 19,889.26 ± 3,345.77 J/mol/K
- Host-Adsorbate energy: -134,610.32 ± 1,654.17 K
- Adsorbate-Adsorbate energy: -2,024.26 ± 51.85 K

### Monte Carlo Performance
- Swap addition moves: 44.43% acceptance rate
- Swap deletion moves: 44.54% acceptance rate
- Reinsertion moves: 33.04% acceptance rate
- Energy drift: Very low (1.31e-08), indicating stable simulation

## Interpretation
The negative enthalpy of adsorption (-17.40 kJ/mol) indicates that methane adsorption on IRMOF-13 is exothermic, meaning energy is released when methane molecules adsorb onto the framework. This is typical for physisorption processes and suggests favorable methane-framework interactions.

## Files Generated
- simulation.input: Main input file
- framework.cif: IRMOF-13 structure
- methane.def: Methane molecule definition
- Various force field and output files
- Energy and density histograms
- Complete simulation output data

## Simulation Quality
The simulation shows excellent convergence with:
- Low energy drift
- Reasonable acceptance rates for all moves
- Stable statistical averages
- Appropriate error bars on results

The calculated adsorption enthalpy of -17.40 ± 0.28 kJ/mol provides a reliable answer to the research question.