RASPA Simulation Setup for Adsorption Enthalpy Calculation
===========================================================

Task: Determine the adsorption enthalpy of n-pentane on IRMOF-13

Given Parameters:
- Helium void fraction: 0.877
- Ideal gas rosenbluth weight for n-pentane: 0.0197439

Steps Completed:

1. Framework Setup:
   - Loaded IRMOF-13 framework using framework loader
   - Generated framework.cif file
   - Recommended unit cells: [2, 2, 1] for cutoff 12.8 Å

2. Molecule Setup:
   - Loaded n-pentane molecule definition
   - Generated pentane.def and associated force field files
   - Files created: pentane.def, pseudo_atoms.def, force_field_mixing_rules.def, force_field.def

3. Simulation Input File (simulation.input):
   - Simulation Type: MonteCarlo (for infinite dilution method)
   - Cycles: 100,000 with 10,000 initialization cycles
   - Framework: IRMOF-13 with unit cells [2, 2, 1]
   - Temperature: 300.0 K
   - Pressure: 0.0 Pa (infinite dilution conditions)
   - Single molecule creation (CreateNumberOfMolecules: 1)
   - MC moves: Translation (50%) and Reinsertion (50%)
   - Applied given parameters: HeliumVoidFraction and IdealGasRosenbluthWeight

Calculation Method:
- Uses infinite dilution approach for adsorption enthalpy
- Formula: ΔH = ⟨U_hg⟩ - ⟨U_h⟩ - ⟨U_g⟩ - RT
- For rigid frameworks: ΔH = (Total_energy - T) × R_gas_constant
- Where Total_energy comes from simulation output

Files Ready for Simulation:
- simulation.input (main input file)
- framework.cif (IRMOF-13 structure)
- pentane.def (n-pentane molecule definition)
- force_field.def (force field parameters)
- pseudo_atoms.def (pseudoatom definitions)
- force_field_mixing_rules.def (mixing rules)

Next Step: Execute the simulation using 'execute raspa' command

Note: The simulation is set up but NOT executed as requested.
