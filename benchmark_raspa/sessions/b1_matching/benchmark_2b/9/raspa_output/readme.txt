HELIUM VOID FRACTION CALCULATION FOR IRMOF-13
==============================================

OBJECTIVE:
Calculate the helium void fraction of IRMOF-13 using RASPA Monte Carlo simulations.

RESULT:
Helium Void Fraction = 0.74254 ± 0.000932 (74.25%)

STEPS PERFORMED:

1. FRAMEWORK LOADING:
   - Loaded IRMOF-13 framework using framework loader
   - Generated framework.cif file
   - Recommended unit cells: [2, 2, 1] for cutoff 12.8 Å

2. MOLECULE LOADING:
   - Loaded helium molecule definition
   - Generated helium.def and associated force field files

3. SIMULATION SETUP:
   - SimulationType: MonteCarlo
   - NumberOfCycles: 25000
   - NumberOfInitializationCycles: 5000
   - Used Widom insertion method (WidomProbability: 1.0)
   - CreateNumberOfMolecules: 0 (no actual molecules, only test insertions)
   - Temperature: 298.0 K
   - Framework unit cells: 2 2 1

4. SIMULATION EXECUTION:
   - Successfully ran RASPA simulation
   - Framework properties: R -3 m space group, hexagonal cell
   - Cell parameters: a=b=24.82 Å, c=56.73 Å, γ=120°

5. RESULTS ANALYSIS:
   - Average Widom Rosenbluth factor: 0.74254 ± 0.000932
   - This represents the helium void fraction (accessible volume fraction)
   - Average volume: 121,087.5 Å³
   - Henry coefficient: 3.99×10⁻⁷ mol/kg/Pa

INTERPRETation:
- IRMOF-13 has a high void fraction of 74.25%
- This indicates excellent porosity for gas storage applications
- The low uncertainty (±0.000932) shows good statistical convergence
- This void fraction is a prerequisite for GCMC adsorption simulations

FILES GENERATED:
- framework.cif (IRMOF-13 structure)
- helium.def (helium molecule definition)
- force_field.def, pseudo_atoms.def (force field parameters)
- simulation.input (RASPA input file)
- Output files with detailed results

NOTE:
This helium void fraction calculation is essential before running gas adsorption isotherms or other framework-based simulations in RASPA.