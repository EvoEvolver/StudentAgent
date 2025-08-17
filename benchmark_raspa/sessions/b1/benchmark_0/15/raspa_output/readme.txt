RASPA Simulation Setup for n-Heptane Adsorption Enthalpy on IRMOF-13
=====================================================================

This simulation setup calculates the adsorption enthalpy of n-heptane on IRMOF-13 framework.

Files Generated:
1. framework.cif - IRMOF-13 framework structure with unit cells [2,2,1]
2. n-heptane.def - n-heptane molecule definition
3. helium.def - helium molecule definition
4. force_field.def - force field parameters
5. pseudo_atoms.def - pseudoatom definitions
6. force_field_mixing_rules.def - mixing rules

Simulation Steps:
1. simulation.input - Helium void fraction calculation using Widom insertion
2. simulation_298K.input - n-heptane adsorption at 298K
3. simulation_273K.input - n-heptane adsorption at 273K

To Calculate Adsorption Enthalpy:
1. First run helium void fraction simulation to get the void fraction value
2. Update the HeliumVoidFraction parameter in the n-heptane simulations
3. Run both temperature simulations
4. Calculate enthalpy from: ΔH = -R * d(ln(K))/d(1/T)
   where K is the Henry coefficient from each simulation

Simulation Parameters:
- Reduced cycles (1000) for faster execution (10% of typical)
- Maximum 32 molecules as requested
- Cutoff: 12.8 Å
- Framework: IRMOF-13 with unit cells [2,2,1]
- Temperatures: 273K and 298K for enthalpy calculation

Note: The helium void fraction is initially set to 0.75 (estimate) but should be updated with the actual calculated value from the first simulation.
