RASPA Henry Coefficient Calculation for n-heptane on IRMOF-13
=============================================================

This simulation determines the Henry coefficient of n-heptane on IRMOF-13 framework.

Steps Performed:

1. FRAMEWORK LOADING:
   - Loaded IRMOF-13 framework using framework loader tool
   - Generated framework.cif file with unit cells [2, 2, 1] for cutoff 12.8 Å

2. MOLECULE LOADING:
   - Loaded n-heptane molecule using molecule loader tool
   - Loaded helium molecule (required for void fraction calculation)
   - Generated .def files, force field, and pseudoatoms files

3. PREREQUISITE CALCULATION:
   - Created input file for helium void fraction calculation (prerequisite for Henry coefficient)
   - Used Monte Carlo simulation with 10,000 cycles (reduced from typical values)
   - Temperature: 298.0 K, Pressure: 1e5 Pa
   - ComputeHeliumVoidFraction: yes

4. HENRY COEFFICIENT CALCULATION:
   - Created input file for Henry coefficient calculation of n-heptane
   - Used Monte Carlo simulation with 10,000 cycles
   - Temperature: 298.0 K, Pressure: 1e5 Pa
   - ComputeHenryCoefficient: yes
   - Used estimated helium void fraction of 0.5 (should be replaced with actual calculated value)

5. SIMULATION PARAMETERS:
   - Simulation type: Monte Carlo
   - Cycles: 10,000 (1/10 of typical values for faster execution)
   - Initialization cycles: 5,000
   - Cutoff: 12.8 Å for both VDW and Coulomb
   - Charge method: Ewald with precision 1e-6

FILES GENERATED:
- framework.cif (IRMOF-13 structure)
- n-heptane.def (n-heptane molecule definition)
- helium.def (helium molecule definition)
- force_field.def (force field parameters)
- pseudo_atoms.def (pseudoatom definitions)
- force_field_mixing_rules.def (mixing rules)
- simulation.input (RASPA input file)

NOTE: The helium void fraction value (0.5) used in the Henry coefficient calculation is an estimate. In a complete workflow, this should be replaced with the actual calculated value from the helium void fraction simulation.

The Henry coefficient will be output in the RASPA results after simulation execution.