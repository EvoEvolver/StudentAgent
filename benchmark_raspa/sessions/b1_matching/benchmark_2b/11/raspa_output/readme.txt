ADSORPTION ENTHALPY CALCULATION: N2 on IRMOF-13
==============================================

OBJECTIVE:
Determine the adsorption enthalpy of N2 on IRMOF-13 using Monte Carlo simulation at infinite dilution conditions.

GIVEN PARAMETERS:
- Framework: IRMOF-13
- Adsorbate: N2 (nitrogen)
- Helium void fraction: 0.877
- Temperature: 300 K
- Conditions: Infinite dilution (External pressure = 0.0 Pa)

SIMULATION SETUP:
================
1. Framework Loading:
   - Loaded IRMOF-13 framework as framework.cif
   - Unit cells used: [2, 2, 1] (ensures >24 Å for 12.8 Å cutoff)

2. Molecule Definition:
   - Loaded N2 molecule definition (nitrogen.def)
   - Force field files generated automatically

3. Simulation Parameters:
   - SimulationType: MonteCarlo
   - NumberOfCycles: 100,000
   - NumberOfInitializationCycles: 10,000
   - CreateNumberOfMolecules: 1 (infinite dilution)
   - ExternalPressure: 0.0 Pa
   - ExternalTemperature: 300.0 K
   - CutOffVDW: 12.8 Å
   - CutOffCoulomb: 12.8 Å
   - HeliumVoidFraction: 0.877

RESULTS:
========
Simulation Output:
- Total energy: -32552.75331 ± 4.60672 K
- Host-Adsorbate energy: -1303.9864 ± 4.60824 K
- Adsorbate-Adsorbate energy: -1.64834 ± 0.01054 K
- Tail-correction energy: -31247.11857 ± 0.00048 K

ADSORPTION ENTHALPY CALCULATION:
===============================
Formula: ΔH = (Total_energy - T) × R_gas_constant
Where:
- Total_energy = -32552.75331 K
- T = 300 K (temperature)
- R_gas_constant = 8.314462618 J/mol/K

Calculation:
ΔH = (-32552.75331 - 300) × 8.314462618/1000
ΔH = -32852.75331 × 0.008314462618
ΔH = -273.4 kJ/mol

FINAL ANSWER:
============
The adsorption enthalpy of N2 on IRMOF-13 at infinite dilution is:
ΔH = -273.4 ± 0.04 kJ/mol

This negative value indicates that N2 adsorption on IRMOF-13 is exothermic, meaning energy is released when N2 molecules adsorb onto the framework surface.

FILES GENERATED:
===============
- framework.cif: IRMOF-13 structure
- nitrogen.def: N2 molecule definition
- force_field.def: Force field parameters
- simulation.input: RASPA input file
- Output files in simulation_1/Output/
- Energy histograms in simulation_1/EnergyHistograms/

NOTES:
======
- Simulation converged successfully with good statistics
- Framework treated as rigid (appropriate for MOFs)
- Infinite dilution achieved using single molecule insertion
- Error propagation included in final result
