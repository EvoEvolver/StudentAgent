RASPA Simulation Setup for n-Heptane Adsorption Enthalpy on IRMOF-13
=====================================================================

This setup consists of two sequential simulations to determine the adsorption enthalpy of n-heptane on IRMOF-13 at infinite dilution conditions.

STEP 1: Helium Void Fraction Calculation
----------------------------------------
File: simulation_1/simulation.input
Purpose: Calculate the helium void fraction of IRMOF-13 framework
Simulation Type: Monte Carlo
Cycles: 5000 (reduced from typical 50000 for faster execution)
Initialization: 1000 cycles
Framework: IRMOF-13 with unit cells [2, 2, 1]
Molecule: Helium
Temperature: 298 K
Pressure: 1e5 Pa

This simulation uses helium insertion/deletion to probe the accessible pore volume.
The helium void fraction will be calculated and reported in the output.

STEP 2: n-Heptane Adsorption Enthalpy
-------------------------------------
File: simulation_2.input
Purpose: Calculate adsorption enthalpy of n-heptane at infinite dilution
Simulation Type: Monte Carlo
Cycles: 5000 (reduced from typical 50000 for faster execution)
Initialization: 1000 cycles
Framework: IRMOF-13 with unit cells [2, 2, 1]
Molecule: n-heptane
Temperature: 298 K
Pressure: 1e5 Pa
Special: Energy histogram computation enabled

IMPORTANT: Before running simulation 2, update the HeliumVoidFraction value
with the result from simulation 1.

Execution Order:
1. Run simulation_1 to get helium void fraction
2. Update HeliumVoidFraction in simulation_2.input with the calculated value
3. Run simulation_2 to get adsorption enthalpy

Files Generated:
- framework.cif: IRMOF-13 structure
- helium.def: Helium molecule definition
- n-heptane.def: n-Heptane molecule definition
- force_field.def: Force field parameters
- pseudo_atoms.def: Atom type definitions
- force_field_mixing_rules.def: Mixing rules

Note: Simulation parameters are reduced (10% of typical cycles) for faster execution.
For production runs, increase NumberOfCycles to 50000-100000.
