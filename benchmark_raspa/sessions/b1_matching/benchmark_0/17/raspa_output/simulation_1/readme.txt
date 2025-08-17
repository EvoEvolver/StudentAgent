HENRY COEFFICIENT CALCULATION FOR CO2 ON IRMOF-13
=================================================

This directory contains all files needed to calculate the Henry coefficient of CO2 on IRMOF-13 using RASPA.

STEPS PERFORMED:

1. FRAMEWORK SETUP:
   - Loaded IRMOF-13 framework using framework loader
   - Generated framework.cif with unit cell dimensions: 24.8 x 24.8 x 56.7 Å
   - Recommended unit cells for 12.8 Å cutoff: [2, 2, 1]

2. MOLECULE DEFINITIONS:
   - Created CO2.def manually (linear triatomic molecule, C-O bond length 1.16 Å)
   - Loaded helium.def using molecule loader (needed for void fraction calculation)
   - Generated methane.def (testing purposes)

3. FORCE FIELD FILES:
   - Generated force_field.def, force_field_mixing_rules.def, pseudo_atoms.def
   - All using local force field parameters

4. SIMULATION SETUP:
   - Created two simulation input files following prerequisite workflow:
   
   a) simulation.input - HELIUM VOID FRACTION CALCULATION (PREREQUISITE):
      - Monte Carlo simulation
      - 2000 cycles (reduced from typical 20000 for speed)
      - 500 initialization cycles
      - Temperature: 298 K
      - Pressure: 1e5 Pa
      - Framework: IRMOF-13 with unit cells [2, 2, 1]
   
   b) henry_simulation.input - HENRY COEFFICIENT CALCULATION (MAIN):
      - Monte Carlo simulation
      - 5000 cycles (reduced from typical 50000 for speed)
      - 1000 initialization cycles
      - Temperature: 298 K
      - Low pressure: 1e3 Pa (infinite dilution conditions)
      - Framework: IRMOF-13 with unit cells [2, 2, 1]
      - Component: CO2 with all Monte Carlo moves enabled
      - ComputeHenryCoefficients: yes

EXECUTION ORDER:
1. First run: simulation.input (helium void fraction calculation)
2. Extract helium void fraction value from output
3. Update henry_simulation.input with the calculated helium void fraction
4. Run: henry_simulation.input (Henry coefficient calculation)
5. Extract Henry coefficient from output files

NOTES:
- Simulation cycles reduced to 1/10 of typical values for faster execution
- Used infinite dilution conditions (low pressure) for Henry coefficient
- All cutoffs set to 12.8 Å as recommended
- Ewald summation used for electrostatic interactions
- Local force field parameters used throughout

FILES INCLUDED:
- framework.cif: IRMOF-13 crystal structure
- CO2.def: Carbon dioxide molecule definition
- helium.def: Helium molecule definition
- simulation.input: Helium void fraction calculation
- henry_simulation.input: Henry coefficient calculation
- force_field.def: Force field parameters
- force_field_mixing_rules.def: Mixing rules
- pseudo_atoms.def: Pseudoatom definitions
- readme.txt: This documentation file
