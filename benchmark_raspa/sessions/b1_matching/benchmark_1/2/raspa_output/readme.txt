# IRMOF-13 Surface Area Calculation Setup

## Overview
This setup determines the surface area of IRMOF-13 framework using RASPA molecular simulation.

## Steps Performed:

### 1. Framework Loading
- Loaded IRMOF-13 framework structure (framework.cif)
- Unit cell dimensions: a=24.82Å, b=24.82Å, c=56.73Å
- Space group: R -3 m (trigonal)
- Recommended unit cells for 12.8Å cutoff: [2,2,1]

### 2. Molecule Setup
- Generated helium probe molecule definition (helium.def)
- Created force field files (force_field.def, pseudo_atoms.def, force_field_mixing_rules.def)

### 3. Helium Void Fraction Calculation (Prerequisite)
- **Purpose**: Calculate accessible void space in framework
- **Method**: Monte Carlo with Widom insertions
- **Parameters**:
  - Cycles: 1000 (reduced from typical 10,000 for speed)
  - Initialization: 500 cycles
  - Temperature: 298K
  - Pressure: 1e5 Pa
  - Component: Helium with WidomProbability 1.0

### 4. Surface Area Calculation
- **Method**: Geometric surface area using probe molecule approach
- **Parameters**:
  - ComputeSurfaceArea: yes
  - SurfaceAreaSamplingPointsPerSphere: 100
  - SurfaceAreaProbeDistance: Minimum (uses 2^(1/6)σ ≈ 1.12246σ)
  - SurfaceAreaProbeAtom: He
  - Component: Helium with SurfaceAreaProbability 1.0

### 5. Key Files Created:
- `framework.cif`: IRMOF-13 structure file
- `helium.def`: Helium probe molecule definition
- `simulation.input`: Simulation parameters
- Force field files: `force_field.def`, `pseudo_atoms.def`, `force_field_mixing_rules.def`

### 6. Expected Output:
- Surface area in [m²/cm³] and [m²/g] units
- Helium void fraction value
- Framework density and volume information

## Notes:
- Simulation cycles reduced to 1/10 of typical values for faster execution
- Unit cells [2,2,1] chosen to satisfy minimum 2×cutoff requirement (25.6Å)
- Helium void fraction is essential prerequisite for accurate surface area calculation
- Results will be found in Output/System_0/ directory

## Framework Properties:
- **IRMOF-13**: Isoreticular Metal-Organic Framework
- **Citation**: Eddaoudi et al., Science 295, 469-472 (2002)
- **Application**: Methane storage and gas separation