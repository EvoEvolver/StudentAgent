RASPA Simulation Setup for Comparing Adsorption Enthalpies of N2 and Methane on IRMOF-13
======================================================================================

Objective: Compare adsorption enthalpies of nitrogen (N2) and methane (CH4) on IRMOF-13 framework

Steps Performed:

1. FRAMEWORK SETUP:
   - Loaded IRMOF-13 framework using framework loader tool
   - Generated framework.cif file with unit cells [2, 2, 1] for cutoff 12.8 Å

2. MOLECULE SETUP:
   - Generated molecule definitions for nitrogen and methane using molecule loader
   - Created nitrogen.def and methane.def files
   - Generated corresponding force field parameters (force_field.def, force_field_mixing_rules.def)
   - Created pseudo_atoms.def file for atomic interactions

3. SIMULATION INPUT FILES:
   - Created nitrogen_simulation.input for N2 adsorption study
   - Created methane_simulation.input for CH4 adsorption study
   - Both simulations use identical parameters for fair comparison:
     * Monte Carlo simulation type
     * 5,000 production cycles (reduced from typical 50,000+ for speed)
     * 2,000 initialization cycles
     * Temperature: 298 K (room temperature)
     * Pressure: 1e4 Pa (low pressure for Henry's law regime)
     * Framework: IRMOF-13 with helium void fraction 0.7

4. SIMULATION PARAMETERS:
   - Forcefield: local (uses generated force field files)
   - Electrostatics: Ewald summation with 1e-6 precision
   - Cutoffs: 12.8 Å for both VDW and Coulomb interactions
   - Energy and molecule number histograms enabled for analysis

5. FILES GENERATED:
   - framework.cif (IRMOF-13 structure)
   - nitrogen.def, methane.def (molecule definitions)
   - force_field.def, force_field_mixing_rules.def (interaction parameters)
   - pseudo_atoms.def (atomic parameters)
   - nitrogen_simulation.input (N2 simulation setup)
   - methane_simulation.input (CH4 simulation setup)

Next Steps:
- Run both simulations using RASPA
- Extract adsorption enthalpies from output files
- Compare the values to determine which molecule has stronger interaction with IRMOF-13

Note: Simulation cycles were reduced to 1/10 of typical values for faster execution while maintaining qualitative comparison validity.
