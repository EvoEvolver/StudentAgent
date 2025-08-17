# RASPA Simulation Setup: Adsorption Enthalpy Comparison
# n-heptane vs CO2 on IRMOF-13

## Objective:
Compare adsorption enthalpies of n-heptane and CO2 on IRMOF-13 framework
Given: Helium void fraction = 0.877

## Files Created:
1. framework.cif - IRMOF-13 framework structure
2. n-heptane.def - n-heptane molecule definition (auto-generated)
3. CO2.def - CO2 molecule definition (manually created)
4. simulation.input - RASPA input file for GCMC simulation

## Simulation Setup Details:

### Framework:
- Material: IRMOF-13
- Unit cells: [2, 2, 1] (ensures >24Å box size for 12.8Å cutoff)
- Helium void fraction: 0.877 (provided)
- Temperature: 298.0 K
- Pressure range: 1e4 to 1e6 Pa (5 pressure points)

### Simulation Parameters:
- Type: Grand Canonical Monte Carlo (GCMC)
- Total cycles: 500 (reduced from typical 5000+ for speed)
- Initialization cycles: 100 (reduced from typical 1000+)
- Print frequency: every 50 cycles
- Cutoffs: 12.8 Å for both VDW and Coulomb
- Charge method: Ewald with 1e-6 precision

### Components:
1. n-heptane: Linear alkane molecule
2. CO2: Linear molecule (TraPPE model)

### Monte Carlo Moves (for both components):
- Swap moves: 1.0 probability (essential for GCMC)
- Translation: 1.0 probability
- Rotation: 1.0 probability  
- Reinsertion: 1.0 probability

### Output Analysis:
- Energy histograms enabled
- Number of molecules histograms enabled
- Adsorption enthalpies will be calculated using fluctuation formulas
- Results will show both absolute and excess adsorption

## Next Steps:
1. Execute simulation using 'execute raspa' command
2. Parse output files to extract adsorption enthalpies
3. Compare enthalpy values between n-heptane and CO2

## Notes:
- Simulation parameters reduced for faster execution as requested
- Maximum 8 molecules limit maintained
- CO2.def created manually due to molecule loader limitations
- Framework loaded successfully with appropriate unit cell sizing
