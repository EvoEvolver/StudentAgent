Henry Coefficient Calculation: n-heptane on IRMOF-13
=====================================================

This simulation determines the Henry coefficient of n-heptane adsorption on IRMOF-13 framework.

STEPS PERFORMED:

1. PREREQUISITE SIMULATIONS:
   - Helium void fraction calculation (MANDATORY)
   - Ideal Rosenbluth weight calculation (if needed)

2. MAIN SIMULATION SETUP:
   - Framework: IRMOF-13 loaded as framework.cif
   - Molecule: n-heptane defined with force field parameters
   - Simulation: Monte Carlo at infinite dilution conditions

3. REDUCED PARAMETERS (for speed):
   - Cycles: 1/10 of typical values (1000-5000 instead of 50000+)
   - Maximum 8 molecules
   - Temperature: 298K
   - Low pressure for infinite dilution

4. RESULTS:
   - Henry coefficient in mol/(kg·Pa)
   - Quantifies n-heptane adsorption affinity

IMPORTANT NOTES:
- Helium void fraction from prerequisite is used in main calculation
- Reduced parameters trade accuracy for simulation speed
- Henry coefficient is calculated at infinite dilution conditions
- Framework-based Monte Carlo simulation type used

APPLICATIONS:
- Gas separation process design
- Storage capacity estimation
- MOF material screening
