# Henry Coefficient Determination: n-pentane on IRMOF-13

## Complete Procedure Using Available Tools

### Overview
Determining Henry coefficient requires a two-step simulation approach:
1. Prerequisite: Calculate ideal gas Rosenbluth weight for n-pentane
2. Main simulation: Calculate Henry coefficient using IRMOF-13 framework

### Step-by-Step Procedure

#### Step 1: Framework Preparation
- Use `framework loader` tool with parameter: framework_name="IRMOF-13"
- This generates framework.cif file for the IRMOF-13 structure
- Framework automatically sized with proper unit cells (>2×cutoff = 25.6Å)

#### Step 2: Molecule Preparation  
- Use `Molecule loader` tool with parameter: molecule_names=["n-pentane"]
- This generates:
  - n-pentane.def file (molecular geometry)
  - Force field parameters
  - Pseudoatoms files

#### Step 3: Prerequisite - HeliumVoidFraction Calculation
- Use `input_file` tool to create simulation for helium void fraction
- Parameters: Monte Carlo, helium molecule, IRMOF-13 framework
- Use `execute raspa` tool to run simulation
- Use `output_parser` tool to extract void fraction value
- Record this value for main simulations

#### Step 4: Prerequisite - Ideal Gas Rosenbluth Weight
- Use `input_file` tool with these specifications:
  - SimulationType: MonteCarlo
  - Empty box (30×30×30 Angstrom, no framework)
  - WidomProbability: 1.0
  - CreateNumberOfMolecules: 0
  - Component: n-pentane
  - Temperature: [desired temperature, e.g., 298K]
- Use `execute raspa` tool to run simulation
- Use `output_parser` tool to extract "Average Widom Rosenbluth factor"
- Record this value for Henry coefficient simulation

#### Step 5: Main Henry Coefficient Simulation
- Use `input_file` tool with these specifications:
  - SimulationType: MonteCarlo
  - Framework: IRMOF-13
  - HeliumVoidFraction: [value from Step 3]
  - WidomProbability: 1.0
  - CreateNumberOfMolecules: 0
  - Component: n-pentane with IdealGasRosenbluthWeight: [value from Step 4]
  - Temperature: [same as Step 4]
- Use `execute raspa` tool to run simulation
- Use `output_parser` tool to extract "Average Henry coefficient [mol/kg/Pa]"

#### Step 6: Results Analysis
- Use `read_file` tool to examine detailed output if needed
- Henry coefficient will be reported in units of [mol/kg/Pa]
- Error estimates provided for statistical accuracy

### Key Technical Requirements
- Both simulations use Widom insertion method (no actual molecule insertion)
- Monte Carlo simulation type required
- Proper force field and pseudoatoms files essential
- Temperature consistency between prerequisite and main simulation
- Framework unit cells automatically sized >25.6Å for 12.8Å cutoff

### Tools Used in Order
1. `framework loader` - Load IRMOF-13 structure
2. `Molecule loader` - Generate n-pentane files
3. `input_file` - Create helium void fraction simulation
4. `execute raspa` - Run void fraction calculation
5. `output_parser` - Extract void fraction
6. `input_file` - Create Rosenbluth weight simulation
7. `execute raspa` - Run Rosenbluth calculation
8. `output_parser` - Extract Rosenbluth weight
9. `input_file` - Create Henry coefficient simulation
10. `execute raspa` - Run Henry coefficient calculation
11. `output_parser` - Extract final Henry coefficient
12. `read_file` - Additional analysis if needed

### Expected Output
Final result: Henry coefficient of n-pentane on IRMOF-13 in [mol/kg/Pa] with statistical error estimates.

This coefficient represents the adsorption behavior at infinite dilution conditions and characterizes the initial slope of the adsorption isotherm.