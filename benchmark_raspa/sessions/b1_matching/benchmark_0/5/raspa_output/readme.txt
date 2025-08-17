ADSORPTION ENTHALPY COMPARISON: CO2 vs METHANE on IRMOF-13

This project compares the adsorption enthalpies of CO2 and methane on IRMOF-13 framework using RASPA molecular simulations.

STEPS EXPLANATION (without execution):

1. FRAMEWORK PREPARATION
   - Load IRMOF-13 framework using framework loader tool
   - This generates the framework.cif file with crystal structure

2. MOLECULE PREPARATION
   - Load CO2 and methane molecules using molecule loader tool
   - This generates .def files, force field parameters, and pseudoatoms files

3. PREREQUISITE SIMULATIONS
   - Calculate helium void fraction for IRMOF-13 (required for adsorption studies)
   - Calculate ideal Rosenbluth weights for both molecules if needed

4. ADSORPTION SIMULATIONS
   - Run Monte Carlo adsorption simulations for both CO2 and methane
   - Perform simulations at multiple temperatures to calculate temperature dependence
   - Use reduced simulation cycles (1/10 or less) and max 8 molecules for speed
   - Calculate adsorption isotherms and thermodynamic properties

5. ENTHALPY CALCULATION
   - Extract total energy and adsorption data from simulation outputs
   - Calculate adsorption enthalpy from temperature-dependent properties
   - Compare enthalpies between CO2 and methane

6. ANALYSIS AND COMPARISON
   - Parse output files using output parser tool
   - Compare adsorption enthalpies and discuss differences
   - Analyze molecular interactions and framework affinity

NOTE: All simulations use accelerated parameters for demonstration purposes.
