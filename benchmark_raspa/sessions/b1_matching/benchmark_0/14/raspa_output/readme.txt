ADSORPTION ENTHALPY CALCULATION FOR N-HEXANE ON IRMOF-13
=========================================================

Objective: Determine the adsorption enthalpy of n-hexane on IRMOF-13 using RASPA molecular simulations.

Methodology:
-----------
Adsorption enthalpy is calculated from the temperature dependence of Henry coefficients using the van't Hoff equation:
d(ln K_H)/d(1/T) = -ΔH_ads/R

Where:
- K_H = Henry coefficient
- T = Temperature (K)
- ΔH_ads = Adsorption enthalpy (kJ/mol)
- R = Gas constant (8.314 J/mol·K)

Steps Performed:
---------------
1. Framework Setup:
   - Loaded IRMOF-13 framework using framework loader
   - Generated framework.cif file with unit cells [2, 2, 1] for cutoff 12.8 Å

2. Molecule Setup:
   - Loaded n-hexane molecule using molecule loader
   - Generated n-hexane.def, force field files, and pseudoatoms files
   - Also loaded helium for void fraction calculation (prerequisite)

3. Prerequisite Simulation:
   - Created helium void fraction simulation (required before Henry coefficient calculations)
   - Used Monte Carlo simulation with 1000 cycles (reduced for speed)

4. Henry Coefficient Simulations:
   - Created three simulations at different temperatures:
     * 298K (simulation.input in simulation_1/)
     * 318K (simulation_318K.input)
     * 338K (simulation_338K.input)
   - Each simulation uses:
     * Monte Carlo simulation type
     * 1000 production cycles + 500 initialization cycles
     * Widom insertion method (WidomProbability = 1.0)
     * ComputeHenryCoefficients = yes
     * Reduced cycles for faster computation as instructed

5. Analysis Procedure:
   - Run all three simulations to obtain Henry coefficients at each temperature
   - Plot ln(K_H) vs 1/T to get a linear relationship
   - Calculate slope = -ΔH_ads/R
   - Convert to adsorption enthalpy: ΔH_ads = -slope × R

Simulation Parameters:
---------------------
- Framework: IRMOF-13 (2×2×1 unit cells)
- Molecule: n-hexane
- Temperatures: 298K, 318K, 338K
- Pressure: 1×10^5 Pa
- Cutoffs: 12.8 Å (VDW and Coulomb)
- Cycles: 1000 (production) + 500 (initialization)
- Method: Widom insertion for Henry coefficients

Files Generated:
---------------
- framework.cif: IRMOF-13 structure
- n-hexane.def: n-hexane molecule definition
- helium.def: helium molecule definition
- force_field.def, force_field_mixing_rules.def: Force field parameters
- pseudo_atoms.def: Pseudoatom definitions
- simulation.input: 298K Henry coefficient simulation
- simulation_318K.input: 318K Henry coefficient simulation
- simulation_338K.input: 338K Henry coefficient simulation

Next Steps:
----------
1. Execute all three RASPA simulations
2. Extract Henry coefficients from output files
3. Apply van't Hoff analysis to calculate adsorption enthalpy
4. Report final ΔH_ads value in kJ/mol

Note: Simulation cycles were reduced to 1/10 of typical values for faster computation as requested.
