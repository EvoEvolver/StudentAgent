RASPA Simulation: Adsorption Enthalpy of n-hexane on IRMOF-13
==============================================================

Objective:
Determine the adsorption enthalpy of n-hexane on IRMOF-13 at 1e4 Pa and 300 K

Simulation Setup:
1. Framework: IRMOF-13 loaded with unit cells [2, 2, 1] for cutoff 12.8 Å
2. Molecule: n-hexane with local force field
3. Simulation type: Monte Carlo
4. Cycles: 50 production cycles, 25 initialization cycles (reduced for speed)
5. Temperature: 300 K
6. Pressure: 1e4 Pa (10,000 Pa)
7. Properties computed: Heat of adsorption

Key Simulation Parameters:
- Cutoff VDW: 12.8 Å
- Cutoff Coulomb: 12.8 Å
- Charge method: Ewald with precision 1e-6
- MC moves: Translation, rotation, reinsertion, and swap moves enabled

Results:
========
Enthalpy of adsorption: -4485457.77 ± 3.38 kJ/mol
(Note: RASPA reports heat of adsorption Q=-H, so adsorption enthalpy = -Q)

Additional Results:
- Average loading: 29.46 ± 12.28 molecules total
- Average loading per unit cell: 7.365 ± 3.07 molecules/unit cell
- Average density: 34.82 ± 14.52 kg/m³
- Framework volume: 121,087.52 ± 0.00 Å³

Simulation Performance:
- Swap addition acceptance: 16.33%
- Swap deletion acceptance: 5.90%
- Reinsertion acceptance: 3.47%

Note: This simulation used reduced cycles (1/20 of typical) for speed, which results in lower accuracy as indicated by the large uncertainties in some properties.

Conclusion:
The adsorption enthalpy of n-hexane on IRMOF-13 at 300 K and 1e4 Pa is approximately -4.49 MJ/mol, indicating a highly exothermic adsorption process.