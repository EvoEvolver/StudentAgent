# RASPA Simulation: Adsorption Enthalpy Comparison of CO2 vs Methane on IRMOF-13

## Objective
Compare adsorption enthalpies of CO2 and methane on IRMOF-13 framework using Grand Canonical Monte Carlo (GCMC) simulations.

## Simulation Setup
- Framework: IRMOF-13 (unit cells: 2×2×1)
- Helium void fraction: 0.877 (provided)
- Temperature: 298.0 K
- Pressure range: 1×10⁴ to 1×10⁶ Pa (5 pressure points)
- Simulation type: Monte Carlo (GCMC)
- Cycles: 1000 production + 100 initialization
- Force field: Local (RASPA default)
- Cutoffs: 12.8 Å (VDW and Coulomb)

## Results

### CO2 Adsorption
- **Enthalpy of adsorption: -33.38 ± 2.72 kJ/mol**
- Temperature equivalent: -4014.8 ± 327.5 K
- Average loading at highest pressure: 947.2 ± 3.8 molecules
- Swap acceptance rate: ~0.09% (low due to strong interactions)

### Methane Adsorption  
- **Enthalpy of adsorption: -6.14 ± 0.25 kJ/mol**
- Temperature equivalent: -738.7 ± 30.2 K
- Average loading at highest pressure: 17.5 ± 0.6 molecules
- Swap acceptance rate: ~64% (higher due to weaker interactions)

## Key Findings
1. **CO2 binds ~5.4 times stronger** than methane to IRMOF-13
2. CO2 shows much higher loading capacity (947 vs 18 molecules)
3. The stronger CO2-framework interactions result in lower swap acceptance rates
4. Both simulations showed acceptable energy drift (<1×10⁻⁷ K)

## Simulation Quality
- Energy drift within acceptable limits for both simulations
- Statistical errors reasonable given reduced cycle count
- Swap move performance indicates proper equilibration

## Files Generated
- simulation_1/: CO2 simulation results
- simulation_2/: Methane simulation results
- Output data files contain detailed thermodynamic properties
- Energy and density histograms available for analysis

## Conclusion
IRMOF-13 shows strong selectivity for CO2 over methane, with CO2 having significantly higher adsorption enthalpy (-33.4 vs -6.1 kJ/mol) and loading capacity. This makes IRMOF-13 a promising candidate for CO2 capture applications.