# IRMOF-13 Surface Area Calculation Setup

This directory contains all necessary files to determine the surface area of IRMOF-13 using RASPA.

## Files Generated:

1. **framework.cif** - IRMOF-13 framework structure
   - Unit cells: [2, 2, 1] (optimized for 12.8 Å cutoff)

2. **Argon.def** - Argon probe molecule definition
   - Used as the probe atom for surface area measurement

3. **simulation.input** - Main simulation input file
   - Simulation type: Monte Carlo
   - Surface area calculation enabled
   - 10,000 cycles with 5,000 initialization cycles

4. **force_field.def** - Force field parameters
5. **force_field_mixing_rules.def** - Mixing rules
6. **pseudo_atoms.def** - Pseudoatom definitions

## Simulation Setup Details:

### Surface Area Parameters:
- ComputeSurfaceArea: yes
- SurfaceAreaSamplingPointsPerSphere: 1000 (good balance of accuracy/speed)
- SurfaceAreaProbeDistance: Minimum (uses 2^(1/6)σ ≈ 1.12246σ)
- SurfaceAreaProbeAtom: Argon

### Framework Configuration:
- Framework: IRMOF-13
- Temperature: 298.0 K
- Unit cells: 2 2 1 (required for 12.8 Å cutoff)

### Component Setup:
- Probe molecule: Argon
- SurfaceAreaProbability: 1.0
- CreateNumberOfMolecules: 0 (no actual molecules created)

## Steps Performed:

1. **Framework Loading**: Loaded IRMOF-13 framework with appropriate unit cell dimensions
2. **Molecule Loading**: Generated Argon probe molecule definition files
3. **Input File Creation**: Created simulation.input with surface area calculation parameters
4. **Force Field Setup**: Generated all necessary force field and pseudoatom files

## Expected Output:

The simulation will calculate the geometric surface area of IRMOF-13 and report results in:
- m²/cm³ (surface area per unit volume)
- m²/g (specific surface area per unit mass)

## To Execute:

Run the simulation using the execute raspa command. The surface area results will be available in the output files.

## Notes:

- This is a geometric surface area calculation based purely on framework structure
- No prerequisites (like helium void fraction) are required
- The choice of 'Minimum' distance criteria provides conservative surface area estimates
- Results can be compared with experimental BET surface area measurements
