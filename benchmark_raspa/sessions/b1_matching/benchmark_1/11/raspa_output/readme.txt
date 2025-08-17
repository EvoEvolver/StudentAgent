# RASPA Simulation Setup: Adsorption Enthalpy Comparison of N2 and Methane on IRMOF-13

## Objective
Compare the adsorption enthalpies of nitrogen (N2) and methane on IRMOF-13 framework with given helium void fraction of 0.877.

## Files Generated
1. **framework.cif** - IRMOF-13 framework structure (unit cells: 2x2x1 for 12.8Å cutoff)
2. **nitrogen.def** - Nitrogen molecule definition
3. **methane.def** - Methane molecule definition
4. **force_field.def** - Force field parameters
5. **force_field_mixing_rules.def** - Mixing rules for interactions
6. **pseudo_atoms.def** - Pseudoatom definitions
7. **simulation.input** - Main RASPA input file

## Simulation Setup Details

### Simulation Type
- **Grand Canonical Monte Carlo (GCMC)** - Required for adsorption enthalpy calculations
- Allows particle insertion/deletion at constant fugacity
- SwapProbability = 1.0 for both components

### Simulation Parameters (Accelerated)
- NumberOfCycles: 1000 (reduced from typical 10000+ for faster execution)
- NumberOfInitializationCycles: 100 (equilibration period)
- PrintEvery: 100 (output frequency)

### Framework Specifications
- Framework: IRMOF-13
- Unit Cells: 2x2x1 (ensures >24Å perpendicular lengths for 12.8Å cutoff)
- HeliumVoidFraction: 0.877 (given)
- Temperature: 298K (room temperature)
- Pressure Range: 1e4 to 1e6 Pa (0.1 to 10 bar)

### Components
1. **Nitrogen**
   - MoleculeName: nitrogen
   - IdealGasRosenbluthWeight: 1.0 (typical for simple molecules)
   - MC Moves: Translation, Rotation, Reinsertion, Swap

2. **Methane**
   - MoleculeName: methane
   - IdealGasRosenbluthWeight: 1.0 (typical for simple molecules)
   - MC Moves: Translation, Rotation, Reinsertion, Swap

### Force Field Settings
- Forcefield: local
- ChargeMethod: Ewald
- EwaldPrecision: 1e-6
- CutOffVDW: 12.8Å
- CutOffCoulomb: 12.8Å

## Prerequisites Completed
1. ✅ Framework loaded (IRMOF-13)
2. ✅ Molecules loaded (nitrogen, methane)
3. ✅ Force field files generated
4. ✅ Simulation input file created
5. ✅ Helium void fraction provided (0.877)

## Next Steps (NOT EXECUTED)
To run the simulation:
1. Execute RASPA with the generated input file
2. Analyze output for adsorption isotherms
3. Calculate adsorption enthalpies from temperature dependence
4. Compare enthalpies between nitrogen and methane

## Important Notes
- Simulation parameters are reduced for faster execution (1/10 of typical cycles)
- IdealGasRosenbluthWeight values (1.0) are estimates for simple molecules
- For production runs, consider calculating exact Rosenbluth weights via Widom insertion
- Adsorption enthalpy requires temperature-dependent isotherm analysis

## File Structure
```
simulation_1/
├── framework.cif
├── nitrogen.def
├── methane.def
├── force_field.def
├── force_field_mixing_rules.def
├── pseudo_atoms.def
├── simulation.input
└── readme.txt
```

Simulation is ready to execute but NOT run as per instructions.