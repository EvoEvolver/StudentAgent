COMPLETE PROCEDURE: Comparing Adsorption Enthalpies of n-Pentane and Methane on IRMOF-13

=== OVERVIEW ===
This procedure compares adsorption enthalpies of two molecules (n-pentane and methane) on IRMOF-13 framework using Grand Canonical Monte Carlo (GCMC) simulations in RASPA.

=== STEP-BY-STEP PROCEDURE ===

**STEP 1: Framework Preparation**
Tool: framework loader
- Load IRMOF-13 framework: framework_name = "IRMOF-13"
- This generates framework.cif file with proper unit cell dimensions
- Unit cells automatically calculated to be > 2×cutoff (>24Å for 12Å cutoff)

**STEP 2: Molecule Preparation**
Tool: Molecule loader
- Load both molecules: molecule_names = ["n-pentane", "methane"]
- This generates .def files and force field parameters for both molecules

**STEP 3: Prerequisite Calculation - Helium Void Fraction**
Tool: input_file
- Create simulation.input for Widom insertions of Helium on IRMOF-13
- Parameters:
  - SimulationType: MonteCarlo
  - Framework: IRMOF-13
  - Component: Helium with Widom insertions only
  - High number of cycles for statistical accuracy
- Tool: execute raspa
- Tool: output_parser to extract HeliumVoidFraction value

**STEP 4: Prerequisite Calculation - Ideal Gas Rosenbluth Weights**
For each molecule (n-pentane and methane):
Tool: input_file
- Create simulation.input for Widom insertions in empty box
- Parameters:
  - SimulationType: MonteCarlo
  - Box system (not Framework)
  - Large box dimensions (>30Å)
  - Component: target molecule with Widom insertions only
  - Same temperature as main simulation
- Tool: execute raspa
- Tool: output_parser to extract IdealGasRosenbluthWeight

**STEP 5: Main GCMC Simulation - n-Pentane**
Tool: input_file
- Create simulation.input for n-pentane adsorption:
  - SimulationType: MonteCarlo
  - Framework: IRMOF-13
  - HeliumVoidFraction: [value from Step 3]
  - Component 0: n-pentane
    - IdealGasRosenbluthWeight: [value from Step 4]
    - SwapProbability: 1.0
    - Translation/Rotation/Reinsertion: 1.0
  - Pressure range for isotherm
  - High cycle counts for convergence
- Tool: execute raspa
- Tool: output_parser to extract adsorption enthalpy

**STEP 6: Main GCMC Simulation - Methane**
Tool: input_file
- Create simulation.input for methane adsorption:
  - Same framework and conditions as Step 5
  - Component 0: methane
    - IdealGasRosenbluthWeight: [value from Step 4]
    - Same Monte Carlo parameters
- Tool: execute raspa
- Tool: output_parser to extract adsorption enthalpy

**STEP 7: Results Analysis and Comparison**
Tool: read_file (if needed for detailed analysis)
- Compare adsorption enthalpies from both simulations
- Consider statistical error bars
- Analyze temperature and pressure dependencies
- Document which molecule has stronger interaction with IRMOF-13

=== CRITICAL SUCCESS FACTORS ===
1. Prerequisites MUST be calculated first - simulations will fail without them
2. Same temperature and pressure conditions for fair comparison
3. Sufficient cycle counts for statistical convergence
4. Monitor acceptance rates during simulations
5. Framework unit cells must be > 2×cutoff for periodic boundary conditions

=== EXPECTED OUTCOMES ===
- Quantitative adsorption enthalpy values for both molecules
- Direct comparison showing which molecule has stronger framework interaction
- Statistical reliability through error analysis
- Understanding of molecular size and interaction effects on adsorption

=== TOOLS SEQUENCE ===
1. framework loader → IRMOF-13
2. Molecule loader → n-pentane, methane
3. input_file → Helium void fraction calculation
4. execute raspa → Run Helium simulation
5. output_parser → Extract void fraction
6. input_file → n-pentane Rosenbluth weight
7. execute raspa → Run n-pentane ideal gas
8. output_parser → Extract n-pentane weight
9. input_file → methane Rosenbluth weight
10. execute raspa → Run methane ideal gas
11. output_parser → Extract methane weight
12. input_file → n-pentane GCMC on IRMOF-13
13. execute raspa → Run n-pentane adsorption
14. output_parser → Extract n-pentane enthalpy
15. input_file → methane GCMC on IRMOF-13
16. execute raspa → Run methane adsorption
17. output_parser → Extract methane enthalpy
18. Analysis and comparison of results

This procedure ensures accurate and comparable adsorption enthalpy calculations for both molecules on the same framework.