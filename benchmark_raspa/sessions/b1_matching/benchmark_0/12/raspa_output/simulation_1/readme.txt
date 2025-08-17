ADSORPTION ENTHALPY CALCULATION: n-heptane on IRMOF-13 at Infinite Dilution
=============================================================================

OBJECTIVE:
Determine the adsorption enthalpy of n-heptane on IRMOF-13 using RASPA simulation at infinite dilution conditions.

STEPS COMPLETED:

1. FRAMEWORK SETUP:
   - Loaded IRMOF-13 framework using framework loader
   - Generated framework.cif file
   - Unit cells required: [2, 2, 1] for cutoff of 12.8 Angstrom

2. MOLECULE SETUP:
   - Loaded n-heptane molecule using molecule loader
   - Generated n-heptane.def file with molecular geometry and properties
   - Loaded helium molecule for prerequisite void fraction calculation
   - Generated helium.def file

3. FORCE FIELD FILES:
   - Generated force_field.def with interaction parameters
   - Generated force_field_mixing_rules.def for mixing rules
   - Generated pseudo_atoms.def for atomic properties

4. SIMULATION SETUP:
   - Created simulation.input file for adsorption enthalpy calculation
   - Simulation type: Monte Carlo (framework-based)
   - Reduced cycles: 1000 (1/10 of typical amount for faster execution)
   - Temperature: 298.0 K
   - Pressure: 1e5 Pa (1 bar)
   - Infinite dilution: CreateNumberOfMolecules = 0 (insertion/deletion moves)

5. PROPERTIES TO COMPUTE:
   - ComputeHeatOfAdsorption: yes (main objective)
   - ComputeHenryCoefficient: yes (for infinite dilution)
   - ComputeNumberOfMolecules: yes
   - ComputeEnergyHistogram: yes

6. PREREQUISITE CONSIDERATIONS:
   - Helium void fraction assumed as 0.85 (typical for MOFs)
   - In practice, helium void fraction should be calculated first

FILES GENERATED:
- framework.cif: IRMOF-13 crystal structure
- n-heptane.def: n-heptane molecule definition
- helium.def: helium molecule definition
- force_field.def: force field parameters
- force_field_mixing_rules.def: mixing rules
- pseudo_atoms.def: atomic properties
- simulation.input: main simulation input file

NEXT STEPS:
1. Run helium void fraction calculation first (prerequisite)
2. Update simulation.input with calculated void fraction
3. Execute main adsorption enthalpy simulation
4. Parse output files to extract adsorption enthalpy values

NOTE: Simulation cycles reduced to 1000 for faster execution as requested.
For production runs, use 10,000-100,000 cycles for better accuracy.
