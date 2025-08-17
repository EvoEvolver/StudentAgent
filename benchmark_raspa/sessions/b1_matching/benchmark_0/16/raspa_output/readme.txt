ADSORPTION ENTHALPY COMPARISON: n-PENTANE vs n-HEXANE on IRMOF-13
================================================================

OBJECTIVE:
Compare the adsorption enthalpies of n-pentane and n-hexane on IRMOF-13 framework.

STEPS COMPLETED:

1. FRAMEWORK SETUP:
   - Loaded IRMOF-13 framework using framework loader
   - Generated framework.cif file with recommended unit cells [2, 2, 1] for 12.8 Å cutoff

2. MOLECULE SETUP:
   - Loaded n-pentane and n-hexane molecules using molecule loader
   - Generated molecular definition files (.def), force field parameters, and pseudoatoms
   - Files created: pentane.def, n-hexane.def, force_field.def, pseudo_atoms.def, force_field_mixing_rules.def

3. SIMULATION INPUT FILES:
   Created 4 separate input files for temperature-dependent adsorption studies:
   - pentane_298K.input: n-pentane adsorption at 298.15 K
   - pentane_273K.input: n-pentane adsorption at 273.15 K  
   - hexane_298K.input: n-hexane adsorption at 298.15 K
   - hexane_273K.input: n-hexane adsorption at 273.15 K

4. SIMULATION PARAMETERS:
   - Simulation Type: Monte Carlo
   - Cycles: 1000 (reduced for faster computation)
   - Initialization Cycles: 500
   - Framework: IRMOF-13 with helium void fraction 0.29
   - Pressure range: 1e4 to 1e6 Pa (5 pressure points)
   - Temperatures: 273.15 K and 298.15 K
   - Cutoffs: 12.8 Å for both VDW and Coulomb interactions

5. NEXT STEPS (TO BE COMPLETED):
   - Run all 4 simulations
   - Analyze output files for adsorption isotherms
   - Calculate Henry coefficients at both temperatures
   - Determine adsorption enthalpies using van't Hoff equation: ΔH = -R * d(ln K)/d(1/T)
   - Compare enthalpies between n-pentane and n-hexane

FILES GENERATED:
- framework.cif (IRMOF-13 structure)
- pentane.def, n-hexane.def (molecule definitions)
- force_field.def, pseudo_atoms.def (force field parameters)
- 4 input files for temperature-dependent simulations

NOTE: Simulation cycles were reduced to 1/10 of typical values for faster computation as requested.
