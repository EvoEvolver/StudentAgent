RASPA Simulation Setup for Henry Coefficient Calculation
========================================================

Objective: Determine the Henry coefficient of n-heptane on IRMOF-13

Files created:
- framework.cif: IRMOF-13 structure
- helium.def, n-heptane.def: Molecule definitions
- force_field.def, pseudo_atoms.def, force_field_mixing_rules.def: Force field files
- simulation.input: Helium void fraction calculation
- henry_simulation.input: Henry coefficient calculation

Steps performed:
1. Loaded IRMOF-13 framework (minimum unit cells: 2x2x1 for 14Å cutoff)
2. Loaded helium and n-heptane molecules with force field parameters
3. Created simulation.input for helium void fraction calculation:
   - Monte Carlo simulation with 1000 cycles (reduced for speed)
   - Up to 32 helium molecules maximum
   - ComputeHeliumVoidFraction enabled
   - Unit cells: 2x2x2, Temperature: 298K, Pressure: 1e5 Pa

4. Created henry_simulation.input for Henry coefficient calculation:
   - Monte Carlo simulation with Widom insertion method
   - ComputeHenryCoefficients enabled
   - n-heptane as test molecule with WidomProbability = 1.0
   - Placeholder helium void fraction (0.5) - update after first simulation

Execution Instructions:
1. Run simulation.input first to calculate helium void fraction
2. Extract the helium void fraction from output
3. Update HeliumVoidFraction value in henry_simulation.input
4. Run henry_simulation.input to obtain Henry coefficient

Note: All simulation parameters optimized for faster execution as requested.
