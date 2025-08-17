HENRY COEFFICIENT CALCULATION FOR N-PENTANE ON IRMOF-13
=======================================================

Objective: Determine the Henry coefficient of n-pentane on IRMOF-13 at 298K using RASPA simulations

Result: Henry coefficient = 2.07962e-06 ± 8.70781e-08 mol/kg/Pa

METHODOLOGY:
============

The Henry coefficient calculation requires THREE sequential simulations:

1. HELIUM VOID FRACTION CALCULATION (simulation_2)
   - Purpose: Determine accessible void space in IRMOF-13 framework
   - Method: Widom insertion of helium atoms
   - Setup: Framework with WidomProbability 1.0, CreateNumberOfMolecules 0
   - Result: Helium void fraction = 0.327809 ± 0.002598
   - Files: framework.cif, helium.def, force field files

2. IDEAL GAS ROSENBLUTH WEIGHT CALCULATION (simulation_3)
   - Purpose: Calculate reference Rosenbluth weight for n-pentane at 298K
   - Method: Widom insertion in empty box (no framework interactions)
   - Setup: Simple box (30×30×30 Å) with WidomProbability 1.0
   - Result: Ideal gas Rosenbluth weight = 0.0196445 ± 0.000355
   - Files: pentane.def, force field files

3. HENRY COEFFICIENT CALCULATION (simulation_4)
   - Purpose: Calculate final Henry coefficient using Widom insertion
   - Method: Widom insertion of n-pentane in IRMOF-13 framework
   - Setup: Framework with both prerequisites as input parameters
   - Required inputs:
     * HeliumVoidFraction 0.327809
     * IdealGasRosenbluthWeight 0.0196445
   - Result: Henry coefficient = 2.07962e-06 mol/kg/Pa

KEY SIMULATION PARAMETERS:
=========================
- Temperature: 298.0 K
- Framework: IRMOF-13 with unit cells [2, 2, 1]
- Cutoffs: VDW = 12.8 Å, Coulomb = 12.8 Å
- Cycles: 1000 production, 100 initialization (reduced for speed)
- Force field: Local (TraPPE-based)
- Charge method: Ewald summation

FILE STRUCTURE:
==============
simulation_1/ - Initial helium void fraction attempt
simulation_2/ - Successful helium void fraction calculation
simulation_3/ - Ideal gas Rosenbluth weight calculation
simulation_4/ - Final Henry coefficient calculation
simulation_5/ - Current working directory

IMPORTANT INSIGHTS:
==================
1. Henry coefficient calculations require careful sequential workflow
2. Both helium void fraction and ideal gas Rosenbluth weight are mandatory prerequisites
3. Widom insertion method is the standard approach for all three calculations
4. IRMOF-13 shows moderate affinity for n-pentane (Henry coefficient ~2×10⁻⁶ mol/kg/Pa)
5. Error propagation through the three-step process requires careful consideration
6. Framework accessibility (void fraction ~0.33) indicates reasonable porosity

NOTE: Simulations used reduced cycle counts for speed - production calculations should use higher cycle numbers for better statistical accuracy.
