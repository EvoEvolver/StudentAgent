# Complete Procedure: Adsorption Enthalpy of n-heptane on IRMOF-13

## Overview
This procedure determines the adsorption enthalpy of n-heptane on IRMOF-13 using RASPA molecular simulation software through Grand Canonical Monte Carlo (GCMC) simulations.

## Tools Required
1. framework loader - Load IRMOF-13 structure
2. Molecule loader - Generate n-heptane molecular definitions
3. input_file - Create simulation input files
4. execute raspa - Run RASPA simulations
5. output_parser - Analyze simulation results
6. read_file/write_file - File management

## Step-by-Step Procedure

### PHASE 1: Prerequisites (Critical - Must be completed first)

#### Step 1.1: Calculate Helium Void Fraction
1. Use framework loader to load IRMOF-13
2. Use Molecule loader for Helium
3. Create input file for Widom insertion simulation:
   - SimulationType: MonteCarlo
   - Use WidomProbability 1.0
   - Framework: IRMOF-13
   - Component: Helium
   - Calculate accessible void space (value 0-1)
4. Execute RASPA simulation
5. Parse output to extract HeliumVoidFraction value

#### Step 1.2: Calculate Ideal Gas Rosenbluth Weight for n-heptane
1. Use Molecule loader for n-heptane
2. Create input file for empty box simulation:
   - SimulationType: MonteCarlo
   - Use Box (not Framework)
   - Use WidomProbability 1.0
   - Component: n-heptane
   - Temperature-dependent calculation
3. Execute RASPA simulation
4. Parse output to extract IdealGasRosenbluthWeight

### PHASE 2: Main GCMC Adsorption Simulation

#### Step 2.1: Setup Framework and Molecules
1. Use framework loader to load IRMOF-13 as framework.cif
2. Use Molecule loader to generate n-heptane definitions
3. Verify unit cells ensure perpendicular lengths > 24Å (2×cutoff)

#### Step 2.2: Create GCMC Input File
Use input_file tool with these critical parameters:
```
SimulationType                MonteCarlo
NumberOfCycles                [sufficient for convergence]
NumberOfInitializationCycles  [equilibration cycles]

Forcefield                    local
ChargeMethod                  Ewald
EwaldPrecision                1e-6
CutOffVDW                     12.8
CutOffCoulomb                 12.8

Framework 0
FrameworkName framework
UnitCells [x] [y] [z]         # >24Å in each direction
HeliumVoidFraction [value]    # From Step 1.1
ExternalTemperature [T]       # Target temperature
ExternalPressure [P1] [P2] ... # Pressure series for isotherm

Component 0 MoleculeName n-heptane
            MoleculeDefinition local
            IdealGasRosenbluthWeight [value]  # From Step 1.2
            TranslationProbability 0.5
            RotationProbability 0.5
            SwapProbability 1.0              # CRITICAL for GCMC
```

#### Step 2.3: Execute Simulation
1. Use execute raspa to run the GCMC simulation
2. Monitor for successful completion
3. Check acceptance rates for swap moves should be similar for insertion/deletion

#### Step 2.4: Extract Results
1. Use output_parser to analyze simulation output
2. Extract adsorption enthalpy values with statistical errors
3. Verify convergence and statistical reliability

### PHASE 3: Analysis and Validation

#### Step 3.1: Results Interpretation
- Adsorption enthalpy calculated automatically using fluctuation formulas
- Results provided in multiple units with error bars
- Distinguish between absolute and excess adsorption

#### Step 3.2: Quality Checks
- Verify similar acceptance rates for swap addition/deletion
- Check statistical convergence
- Validate against experimental data if available

## Critical Requirements
1. **Unit Cells**: Must be > 2×cutoff (24Å) in all directions
2. **SwapProbability**: Must be 1.0 for GCMC
3. **Prerequisites**: HeliumVoidFraction and IdealGasRosenbluthWeight are mandatory
4. **Force Field**: Use local force field with proper pseudoatoms
5. **Temperature Dependence**: Rosenbluth weights are temperature-dependent

## Common Pitfalls
1. Skipping prerequisite calculations leads to incorrect results
2. Insufficient unit cell size causes cutoff errors
3. Missing swap moves prevents proper GCMC sampling
4. Inadequate equilibration affects statistical accuracy

## Expected Outputs
- Adsorption enthalpy of n-heptane on IRMOF-13 (kJ/mol)
- Statistical error bars for reliability assessment
- Adsorption isotherm data
- Simulation convergence metrics

## Notes
- This procedure requires multiple sequential simulations
- Each phase builds on previous results
- Proper validation ensures scientific accuracy
- Results enable understanding of gas-solid interactions in MOF materials