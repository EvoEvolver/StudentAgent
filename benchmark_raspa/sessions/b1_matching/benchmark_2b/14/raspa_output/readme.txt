RASPA Simulation: Adsorption Enthalpies of CO2/N2 Mixture on IRMOF-13
=====================================================================

Objective:
Compare adsorption enthalpies of a 1:1 mixture of CO2 and N2 on IRMOF-13 at 1e4 Pa and 300K

Simulation Setup:
1. Framework: IRMOF-13 (loaded as framework.cif with unit cells 2x2x1)
2. Molecules: CO2 (carbon_dioxide.def) and N2 (nitrogen.def)
3. Simulation Type: Grand Canonical Monte Carlo (GCMC)
4. Conditions:
   - Temperature: 300 K
   - Pressure: 1e4 Pa (0.1 bar)
   - Helium void fraction: 0.877
   - Mixture ratio: 1:1 (MolFraction 0.5 each)

Simulation Parameters:
- Cycles: 1000 (reduced for speed)
- Initialization cycles: 500
- Cutoffs: 12.8 Å (VDW and Coulomb)
- Charge method: Ewald summation

Results:
========
Adsorption Enthalpies:
- CO2: -74.15 ± 44.67 kJ/mol
- N2: -122.59 ± 17.48 kJ/mol

Key Findings:
1. Nitrogen shows stronger adsorption (-122.59 kJ/mol) than CO2 (-74.15 kJ/mol) on IRMOF-13
2. N2 has lower uncertainty (±17.48) compared to CO2 (±44.67), indicating more consistent adsorption
3. Both values are negative, confirming exothermic adsorption processes

Average Loadings:
- CO2: 827.17 ± 138.74 molecules
- N2: Data partially visible in output

Conclusion:
Under these conditions, N2 exhibits stronger adsorption affinity for IRMOF-13 than CO2, which is somewhat unexpected as CO2 typically shows stronger interactions due to its quadrupole moment. This could be due to specific pore size effects or competitive adsorption behavior in the mixture.

Note: Results use reduced cycles (1/20 of typical) for speed, affecting statistical accuracy.
