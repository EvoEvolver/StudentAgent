# RASPA Simulation Setup: Adsorption Enthalpy Comparison
## n-Pentane vs n-Heptane on IRMOF-13

### Overview
This simulation setup compares the adsorption enthalpies of n-pentane and n-heptane on IRMOF-13 framework using Grand Canonical Monte Carlo (GCMC) method in RASPA.

### Prerequisites Provided
- Helium void fraction: 0.877
- Ideal gas Rosenbluth weight for n-pentane: 0.0197439
- Ideal gas Rosenbluth weight for n-heptane: 0.0004450

### Setup Steps Completed

#### Step 1: Framework Loading
- Loaded IRMOF-13 framework using framework loader
- Generated framework.cif file
- Unit cells set to [2, 2, 1] (minimum required for 12.8 Å cutoff)

#### Step 2: Molecule Loading
- Loaded n-pentane molecule definition (pentane.def)
- Loaded n-heptane molecule definition (n-heptane.def)
- Generated corresponding force field files:
  - force_field.def
  - force_field_mixing_rules.def
  - pseudo_atoms.def

#### Step 3: Simulation Input File Creation
- **Simulation Type**: Monte Carlo (GCMC ensemble)
- **Cycles**: 500,000 production + 1,000 initialization
- **Ensemble**: Grand Canonical Monte Carlo (GCMC)
- **SwapProbability**: 1.0 (enables particle insertion/deletion)
- **Temperature**: 298.0 K
- **Pressure Range**: 1e3 to 1e6 Pa (7 pressure points)
- **Properties Computed**:
  - Energy histograms
  - Number of molecules histograms
  - Molecule properties (including adsorption enthalpies)

#### Step 4: Component Configuration
- **Component 0**: n-pentane with IdealGasRosenbluthWeight 0.0197439
- **Component 1**: n-heptane with IdealGasRosenbluthWeight 0.0004450
- Both components configured for GCMC moves (translation, rotation, reinsertion, swap)

### Files Generated
1. framework.cif - IRMOF-13 structure
2. pentane.def - n-pentane molecule definition
3. n-heptane.def - n-heptane molecule definition
4. force_field.def - Force field parameters
5. force_field_mixing_rules.def - Mixing rules
6. pseudo_atoms.def - Pseudoatom definitions
7. simulation.input - Main simulation input file

### Expected Outputs
After running the simulation, RASPA will generate:
- Adsorption isotherms for both molecules
- Adsorption enthalpies with error bars
- Energy and molecule number histograms
- Statistical analysis of the simulation

### Next Steps
To execute the simulation, run RASPA with the generated input file:
```
raspa simulation.input
```

### Notes
- The simulation uses separate components for direct comparison
- Pressure range covers typical adsorption regimes
- High cycle count ensures statistical reliability
- Framework unit cells satisfy minimum size requirements (>24 Å)
- All required prerequisites (void fraction, Rosenbluth weights) are included

### Comparison Analysis
After simulation completion, compare:
1. Adsorption isotherms at different pressures
2. Adsorption enthalpies (more negative = stronger adsorption)
3. Loading capacities at equivalent pressures
4. Statistical uncertainties in the results
