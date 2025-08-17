RASPA Henry Coefficient Calculation for IRMOF-13
=================================================

Objective: Determine Henry coefficients of methane and CO2 on IRMOF-13

Files Created:
1. framework.cif - IRMOF-13 structure (unit cells: 2×2×1 for 12.8 Å cutoff)
2. methane.def - Methane molecule definition (auto-generated)
3. CO2.def - CO2 molecule definition (manually created)
4. simulation_methane.input - RASPA input for methane Henry coefficient
5. simulation_CO2.input - RASPA input for CO2 Henry coefficient
6. readme.txt - This documentation file

Steps Completed:
1. ✅ Loaded IRMOF-13 framework using framework loader tool
2. ✅ Loaded methane molecule using molecule loader tool
3. ✅ Manually created CO2.def file (PubChem loading failed)
4. ✅ Created separate simulation input files for both molecules
5. ✅ Configured Henry coefficient calculations with Widom insertion

Simulation Parameters:
- Temperature: 298 K
- Pressure: 1×10⁵ Pa
- Monte Carlo cycles: 1,000 (reduced as instructed)
- Initialization cycles: 500
- Method: Widom insertion probability for Henry coefficient
- Estimated helium void fraction: 0.75
- Cutoff distances: 12.8 Å (VDW and Coulomb)
- Charge method: Ewald summation

Molecule Details:
Methane (CH₄):
- Critical temperature: 189.6 K
- Critical pressure: 4,465,382 Pa
- Acentric factor: -0.0197
- Structure: Single CH4 atom (flexible)

CO₂:
- Critical temperature: 304.13 K
- Critical pressure: 7,377,300 Pa
- Acentric factor: 0.224
- Structure: Linear C-O-O (rigid, bond length 1.16 Å)

Current Status:
✅ All files ready for Henry coefficient calculations
✅ Both methane and CO2 simulations configured

Next Steps:
1. Execute RASPA simulation for methane Henry coefficient
2. Execute RASPA simulation for CO2 Henry coefficient
3. Parse and analyze output results
4. Compare Henry coefficients between the two gases

Note: Simulation parameters reduced to 1/10 of typical values for faster execution as instructed. Maximum 8 molecules constraint satisfied (using 0 initial molecules for Widom insertion).
