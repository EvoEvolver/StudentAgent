ADSORPTION ENTHALPY CALCULATION FOR N-HEPTANE ON IRMOF-13
=========================================================

Task: Determine the adsorption enthalpy of n-heptane on IRMOF-13 using simulation at infinite dilution

Given Parameters:
- Helium void fraction: 0.877
- Ideal gas Rosenbluth weight for n-heptane: 0.0004450
- Temperature: 298 K

Simulation Setup:
1. Framework: IRMOF-13 loaded as framework.cif with unit cells [2,2,1]
2. Molecule: n-heptane loaded with force field parameters
3. Simulation type: Monte Carlo with Widom insertions
4. Conditions: Infinite dilution (CreateNumberOfMolecules = 1, ExternalPressure = 0.0)
5. Cycles: 1000 production cycles, 500 initialization cycles

Key Results from Simulation:
- Average Host-Adsorbate energy: -3471.98 ± 0.00 K
- Average Widom chemical potential: -6280.48 ± 158.5 K
- Average Henry coefficient: 30.489 ± 16.53 mol/kg/Pa
- Framework modeled as rigid

Enthalpy Calculation:
For infinite dilution conditions with rigid framework:
ΔH_ads = <U_hg> - RT
Where:
- <U_hg> = Host-Guest interaction energy = -3471.98 K
- R = Gas constant = 8.314 J/(mol·K)
- T = Temperature = 298 K

Calculation:
ΔH_ads = (-3471.98 - 298) × 8.314/1000 = -31.35 kJ/mol

FINAL ANSWER:
The adsorption enthalpy of n-heptane on IRMOF-13 at infinite dilution (298 K) is approximately -31.4 kJ/mol.

This negative value indicates favorable (exothermic) adsorption, meaning energy is released when n-heptane molecules adsorb onto the IRMOF-13 framework.

Simulation Files:
- simulation_1/: Initial Widom insertion attempt
- simulation_3/: Successful simulation with enthalpy calculation
- enthalpy_calculation.py: Manual calculation verification

Note: The simulation used reduced cycles (1000 instead of typical 50000+) for speed as requested, which may affect precision but provides the correct order of magnitude for the adsorption enthalpy.
