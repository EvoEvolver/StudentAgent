HENRY COEFFICIENT CALCULATION FOR N-HEXANE AND N-HEPTANE ON IRMOF-13

This simulation determines the Henry coefficients of n-hexane (C6H14) and n-heptane (C7H16) adsorbed on IRMOF-13 framework using RASPA molecular simulation software.

STEPS PERFORMED:

1. FRAMEWORK SETUP:
   - Loaded IRMOF-13.cif file using framework loader tool
   - Generated framework.cif with proper unit cell dimensions (>30A to satisfy 2x cutoff requirement)

2. MOLECULE DEFINITIONS:
   - Generated n-hexane.def file with molecular geometry and properties
   - Generated n-heptane.def file with molecular geometry and properties
   - Created corresponding force field and pseudoatoms files for both molecules

3. PREREQUISITE CALCULATIONS:
   - Calculated helium void fraction for IRMOF-13 (required for Henry coefficient calculations)
   - This provides the accessible pore volume needed for infinite dilution conditions

4. HENRY COEFFICIENT SIMULATIONS:
   - Set up Monte Carlo simulations at infinite dilution conditions
   - Used reduced cycle numbers (1/10 of standard) for faster computation
   - Limited to maximum 8 molecules as specified
   - Applied proper temperature and pressure conditions
   - Used framework-based MC with appropriate MC moves

5. OUTPUT ANALYSIS:
   - Parsed simulation output files to extract Henry coefficient values
   - Henry coefficients provide thermodynamic data for gas-framework interactions
   - Results useful for gas separation and storage applications

IMPORTANT NOTES:
- All simulations use reduced cycle numbers for accelerated computation
- Framework unit cells automatically sized to satisfy cutoff requirements
- Infinite dilution conditions ensure proper Henry coefficient calculation
- Results provide fundamental adsorption data for both alkane molecules on IRMOF-13