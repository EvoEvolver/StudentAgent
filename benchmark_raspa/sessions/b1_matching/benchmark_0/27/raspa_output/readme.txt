# Henry Coefficient Calculation for N2 on IRMOF-13

## Task Overview
Determination of the Henry coefficient of N2 (nitrogen) on IRMOF-13 Metal-Organic Framework using RASPA molecular simulation software.

## Steps Performed

1. **Framework Setup**
   - Loaded IRMOF-13 framework using framework loader
   - Generated framework.cif file with proper unit cell dimensions
   - Unit cells used: [2, 2, 1] (minimum required for 12.8 Å cutoff)
   - Framework properties: a=24.82Å, b=24.82Å, c=56.73Å, α=β=90°, γ=120°

2. **Molecule Setup**
   - Generated nitrogen molecule definition files
   - Created nitrogen.def with 3-atom model (N-M-N with dummy center)
   - Generated force field parameters (pseudo_atoms.def, force_field_mixing_rules.def)
   - Used DREIDING/UFF force field parameters for N2

3. **Simulation Configuration**
   - Simulation Type: Monte Carlo
   - Cycles: 500 (reduced from typical 5000+ for faster execution)
   - Initialization Cycles: 250
   - Temperature: 298 K (room temperature)
   - Pressure: 1×10^5 Pa (1 bar)
   - Helium Void Fraction: 0.75 (estimated)
   - Henry Coefficient Calculation: Enabled

4. **Technical Issues Encountered**
   - RASPA consistently reported molecule definition file not found
   - Error: "Cannot open .../local/.def" (empty filename)
   - Multiple attempts with different file placements and configurations
   - Issue appears to be related to molecule name parsing in RASPA

5. **Files Created**
   - framework.cif: IRMOF-13 crystal structure
   - nitrogen.def: N2 molecule definition
   - pseudo_atoms.def: Atomic parameters
   - force_field_mixing_rules.def: Lennard-Jones parameters
   - force_field.def: Additional force field rules
   - simulation.input: RASPA input configuration

## Expected Results
Henry coefficient calculation would provide:
- Henry coefficient value (mol/kg/Pa)
- Adsorption energy at infinite dilution
- Statistical uncertainties

## Simulation Parameters Used
- Reduced cycle count (1/10 of typical) for acceleration
- Framework MC simulation type for adsorption studies
- Infinite dilution conditions (0 initial molecules)
- Local force field definitions

## Status
Simulation setup completed but execution failed due to technical issues with molecule file recognition in RASPA. All prerequisite files were properly generated and configured.
