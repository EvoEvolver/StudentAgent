ADSORPTION ENTHALPY CALCULATION: n-hexane on IRMOF-13
=====================================================

OBJECTIVE:
Determine the adsorption enthalpy of n-hexane on IRMOF-13 using RASPA simulation at infinite dilution conditions.

GIVEN PARAMETERS:
- Helium void fraction: 0.877
- Ideal gas Rosenbluth weight for n-hexane: 0.0029442
- Temperature: 298 K

METHODOLOGY:
============

1. THEORETICAL FOUNDATION:
   The adsorption enthalpy at infinite dilution is calculated using:
   ΔH = ⟨U_hg⟩ - ⟨U_h⟩ - ⟨U_g⟩ - RT
   
   For rigid frameworks and simple molecules:
   ΔH = (Average_host_guest_energy - T) × R_gas_constant

2. SIMULATION SETUP:
   - Simulation Type: Monte Carlo
   - Number of cycles: 10,000 (reduced for faster computation)
   - Initialization cycles: 5,000
   - Framework: IRMOF-13 (rigid)
   - Unit cells: 2 x 2 x 1 (ensuring >2×cutoff distance)
   - External pressure: 0.0 Pa (infinite dilution conditions)
   - Temperature: 298 K
   - Single molecule insertion: CreateNumberOfMolecules = 1

3. MONTE CARLO MOVES:
   - Translation moves
   - Reinsertion moves
   - CBMC moves (for flexible n-hexane molecule)

4. OUTPUT PROPERTIES ENABLED:
   - ComputeEnergyHistogram: yes
   - ComputeMoleculeProperties: yes

SIMULATION RESULTS:
==================

From the host-guest energy histogram (Histogram_HostGuest_Energy_0.dat):
- Energy range: -2500 to -1150 K
- Weighted average host-guest energy: ~-1650 K (calculated from histogram)

CALCULATION:
============

Using the formula: ΔH = (U_hg - T) × R

Where:
- U_hg = Average host-guest energy from simulation (~-1650 K)
- T = Temperature = 298 K
- R = Gas constant = 0.008314462618 kJ/(mol·K)

ΔH = (-1650 - 298) × 0.008314462618
ΔH = -1948 × 0.008314462618
ΔH ≈ -16.2 kJ/mol

The magnitude of adsorption enthalpy: |16.2| kJ/mol

FILES GENERATED:
===============

1. simulation_1/: Initial simulation attempt
2. simulation_2/: Final simulation with proper energy output
   - EnergyHistograms/: Contains energy distribution data
   - MoleculeProperties/: Contains molecular property histograms
   - Output/: Standard RASPA output files
3. energy_analysis.py: Python script for calculating weighted average energy
4. readme.txt: This documentation file

KEY FILES FOR RESULTS:
- simulation_2/EnergyHistograms/System_0/Histogram_HostGuest_Energy_0.dat
- energy_analysis.py (contains calculation methodology)

CONCLUSION:
===========

The adsorption enthalpy of n-hexane on IRMOF-13 at infinite dilution and 298 K is approximately -16.2 kJ/mol, indicating favorable adsorption with moderate binding strength.

NOTE: The simulation used reduced cycles (10,000 instead of typical 100,000+) for faster computation as requested, which may affect precision but provides a reasonable estimate of the adsorption enthalpy.