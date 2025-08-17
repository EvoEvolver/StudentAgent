ADSORPTION ENTHALPY CALCULATION: n-hexane on IRMOF-13
=======================================================

This simulation determines the adsorption enthalpy of n-hexane on IRMOF-13 using infinite dilution conditions.

METHODOLOGY:
Adsorption enthalpy is calculated from the temperature dependence of Henry coefficients using the van't Hoff equation:
ln(KH) = -ΔHads/(RT) + constant
where KH is the Henry coefficient, ΔHads is the adsorption enthalpy, R is the gas constant, and T is temperature.

STEPS PERFORMED:
1. Framework Setup: Load IRMOF-13 structure (.cif file)
2. Molecule Setup: Generate n-hexane definition files (.def, force field, pseudoatoms)
3. Multiple Temperature Simulations: Run Monte Carlo simulations at 3-4 different temperatures
4. Henry Coefficient Extraction: Parse output files to extract Henry coefficients
5. van't Hoff Analysis: Plot ln(KH) vs 1/T to determine adsorption enthalpy from slope

SIMULATION CONDITIONS:
- Infinite dilution (very low pressure)
- Framework-based Monte Carlo
- Multiple temperatures (typically 250K, 300K, 350K, 400K)
- Reduced cycles for faster computation

NOTE: This is a multi-step process requiring simulations at different temperatures to extract the enthalpy from the temperature dependence of Henry coefficients.