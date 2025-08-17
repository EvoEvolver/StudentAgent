RASPA Simulation Setup: CO2 Adsorption Enthalpy on IRMOF-13
=============================================================

Objective: Determine the adsorption enthalpy of CO2 on IRMOF-13 framework
Given: Helium void fraction = 0.877

SETUP STEPS COMPLETED:

1. FRAMEWORK SETUP
   - Loaded IRMOF-13 framework using framework loader
   - Generated: framework.cif
   - Unit cells: [2, 2, 1] (recommended for 12.8 Å cutoff)

2. MOLECULE SETUP
   - Loaded CO2 molecule definitions using molecule loader
   - Generated files:
     * carbon dioxide.def (molecule definition)
     * force_field.def (force field parameters)
     * force_field_mixing_rules.def (mixing rules)
     * pseudo_atoms.def (pseudoatom definitions)

3. SIMULATION INPUT FILE
   - Created: simulation.input
   - Simulation type: MonteCarlo
   - Cycles: 100,000 (with 10,000 initialization)
   - Single molecule insertion (CreateNumberOfMolecules 1)
   - Infinite dilution conditions (ExternalPressure 0.0)
   - Temperature: 298 K
   - Helium void fraction: 0.877 (as provided)
   - MC moves: Translation and Reinsertion

THEORETICAL BACKGROUND:
- Adsorption enthalpy at infinite dilution: ΔH = ⟨U_hg⟩ - ⟨U_h⟩ - ⟨U_g⟩ - RT
- For rigid framework: ⟨U_h⟩ = 0
- For simple molecules: ⟨U_g⟩ = 0
- Therefore: ΔH = (Total_energy - T) * R_gas_constant

FILES READY FOR SIMULATION:
- framework.cif
- carbon dioxide.def
- force_field.def
- force_field_mixing_rules.def
- pseudo_atoms.def
- simulation.input

NEXT STEPS:
1. Execute RASPA simulation using 'execute raspa' command
2. Parse output files to extract total energy values
3. Calculate adsorption enthalpy using the formula above

NOTE: All prerequisites have been satisfied - framework structure, molecule definitions, force field parameters, and proper simulation setup for enthalpy calculation at infinite dilution conditions.