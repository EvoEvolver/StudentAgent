ADSORPTION ENTHALPY CALCULATION FOR N-PENTANE ON IRMOF-13
==========================================================

TASK: Determine the adsorption enthalpy of n-pentane on IRMOF-13 using simulation at infinite dilution

GIVEN PARAMETERS:
- Helium void fraction: 0.877
- Ideal gas Rosenbluth weight for n-pentane: 0.0197439
- Temperature: 298 K
- Simulation conditions: Infinite dilution (pressure = 0.0 Pa)

STEPS PERFORMED:

1. FRAMEWORK SETUP:
   - Loaded IRMOF-13 framework using framework loader
   - Generated framework.cif file with unit cells [2, 2, 1] for cutoff 12.8 Å

2. MOLECULE SETUP:
   - Loaded n-pentane molecule definition using molecule loader
   - Generated pentane.def and associated force field files

3. SIMULATION CONFIGURATION:
   - Simulation type: Monte Carlo
   - Number of cycles: 1000 (reduced for speed as requested)
   - Initialization cycles: 100
   - Single molecule insertion (CreateNumberOfMolecules 1)
   - External pressure: 0.0 Pa (infinite dilution condition)
   - Temperature: 298 K
   - Used provided helium void fraction: 0.877
   - Used provided ideal gas Rosenbluth weight: 0.0197439

4. SIMULATION EXECUTION:
   - Successfully ran RASPA simulation
   - Generated energy histograms and output files

5. RESULTS ANALYSIS:
   - Host-Guest interaction energy: -1250 K (from energy histogram)
   - This represents the binding energy between n-pentane and IRMOF-13

6. ADSORPTION ENTHALPY CALCULATION:
   For rigid frameworks at infinite dilution:
   ΔH = (Host-Guest_energy - T) × R_gas_constant
   ΔH = (-1250 - 298) × 8.314462618/1000
   ΔH = -1548 × 0.008314462618
   ΔH = -12.87 kJ/mol

FINAL RESULT:
The adsorption enthalpy of n-pentane on IRMOF-13 at infinite dilution is approximately -12.9 kJ/mol.

NOTE: This calculation used reduced cycles (1000 instead of typical 10000+) for speed as requested, which may affect accuracy. The negative value indicates favorable adsorption (exothermic process).

FILES GENERATED:
- framework.cif (IRMOF-13 structure)
- pentane.def (n-pentane molecule definition)
- force_field.def, pseudo_atoms.def, force_field_mixing_rules.def
- simulation.input (RASPA input file)
- Energy histograms and output data files
