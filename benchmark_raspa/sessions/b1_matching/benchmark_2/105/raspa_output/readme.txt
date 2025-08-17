# Henry Coefficient Determination Procedure
## Task: Determine Henry coefficients of n-pentane and N2 on IRMOF-13

### Overview
Henry coefficient calculation requires 4 separate RASPA simulations:
- 2 prerequisite simulations (ideal gas Rosenbluth weights)
- 2 main simulations (Henry coefficient calculations)

### Complete Step-by-Step Procedure:

#### Step 1: Framework Setup
- Tool: framework loader
- Parameter: framework_name = "IRMOF-13"
- Output: framework.cif file with proper unit cell dimensions
- Note: Automatically sized to meet >30Å requirement

#### Step 2: Molecule Definitions
- Tool: Molecule loader
- Parameter: molecule_names = ["n-pentane", "N2"]
- Output: .def files and force field parameters
- Note: n-pentane requires CBMC due to flexibility

#### Step 3: Prerequisite Simulations (Critical!)

##### 3a. n-pentane Ideal Gas Rosenbluth Weight
- Tool: input_file
- Simulation setup:
  * SimulationType: MonteCarlo
  * Empty box (no framework)
  * WidomProbability: 1.0
  * CreateNumberOfMolecules: 0
  * Cycles: ~20,000
- Tool: execute raspa
- Tool: output_parser (extract IdealGasRosenbluthWeight)

##### 3b. N2 Ideal Gas Rosenbluth Weight
- Similar setup as n-pentane
- Simpler molecule (no CBMC needed)
- Extract IdealGasRosenbluthWeight from output

#### Step 4: Main Henry Coefficient Simulations

##### 4a. n-pentane Henry Coefficient on IRMOF-13
- Tool: input_file
- Simulation setup:
  * Framework: IRMOF-13
  * Component: n-pentane with calculated IdealGasRosenbluthWeight
  * WidomProbability: 1.0
  * CreateNumberOfMolecules: 0
  * High cycle count (500,000+)
- Tool: execute raspa
- Tool: output_parser (extract Henry coefficient)

##### 4b. N2 Henry Coefficient on IRMOF-13
- Similar setup with N2 component
- Use N2's IdealGasRosenbluthWeight
- Extract Henry coefficient from output

#### Step 5: Results Analysis
- Henry coefficients in [mol/kg/Pa] units
- Statistical errors and convergence analysis
- Compare values for selectivity calculations

### Critical Requirements:
1. **Prerequisites are mandatory** - simulations fail without IdealGasRosenbluthWeight
2. **Temperature consistency** across all simulations
3. **Sufficient cycles** for statistical convergence
4. **Proper framework sizing** (handled automatically)

### Tools Required:
- framework loader (IRMOF-13 setup)
- Molecule loader (molecule definitions)
- input_file (4 simulation inputs)
- execute raspa (4 simulation runs)
- output_parser (extract results)
- write_file (documentation)

### Expected Outputs:
- n-pentane Henry coefficient on IRMOF-13
- N2 Henry coefficient on IRMOF-13
- Statistical uncertainties for both values
- Foundation data for adsorption isotherm studies

### Applications:
- Gas separation selectivity calculations
- Adsorption capacity predictions
- MOF performance evaluation
- GCMC simulation prerequisites

Note: This procedure ensures accurate Henry coefficient determination through systematic prerequisite calculations and proper simulation methodology.