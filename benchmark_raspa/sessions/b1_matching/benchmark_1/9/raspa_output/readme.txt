RASPA Simulation Setup for Adsorption Enthalpy of n-Pentane on IRMOF-13
========================================================================

Objective:
Determine the adsorption enthalpy of n-pentane on IRMOF-13 framework using Monte Carlo simulation at infinite dilution conditions.

Given Parameters:
- Helium void fraction: 0.877
- Ideal gas Rosenbluth weight for n-pentane: 0.0197439
- Framework: IRMOF-13
- Adsorbate: n-pentane

Simulation Setup Steps:

1. Framework Loading:
   - Loaded IRMOF-13 framework as framework.cif
   - Unit cells: 2 2 1 (minimum required for 12.8 Å cutoff)

2. Molecule Loading:
   - Loaded n-pentane molecule definition (pentane.def)
   - Generated corresponding force field files

3. Simulation Configuration:
   - Simulation Type: Monte Carlo
   - Cycles: 50,000 (reduced from typical 500,000 for faster execution)
   - Initialization Cycles: 5,000
   - Temperature: 298.0 K (room temperature)
   - Pressure: 0.0 Pa (infinite dilution conditions)
   - Single molecule insertion: CreateNumberOfMolecules = 1

4. Key Files Generated:
   - simulation.input: Main simulation input file
   - framework.cif: IRMOF-13 crystal structure
   - pentane.def: n-pentane molecule definition
   - force_field.def: Force field parameters
   - pseudo_atoms.def: Atomic parameters
   - force_field_mixing_rules.def: Mixing rules

Theoretical Background:
Adsorption enthalpy at infinite dilution is calculated using:
ΔH = ⟨U_hg⟩ - ⟨U_h⟩ - ⟨U_g⟩ - RT

For rigid framework and simple molecules:
ΔH = (Total_energy - T) × R_gas_constant

The simulation will output the 'Total energy' which represents ⟨U_hg⟩,
the average energy of the guest molecule inside the host framework.

Execution:
To run the simulation, execute: raspa simulation.input

Expected Output:
The simulation will provide the total energy value needed to calculate
the adsorption enthalpy of n-pentane on IRMOF-13.

Note: This setup uses reduced cycle counts (1/10 of typical values) for
faster execution as requested, which may affect accuracy.