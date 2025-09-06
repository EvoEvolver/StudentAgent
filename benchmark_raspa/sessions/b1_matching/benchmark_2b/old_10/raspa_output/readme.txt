IRMOF-13 Surface Area Calculation using RASPA
==============================================

Objective: Determine the geometric surface area of IRMOF-13 framework

Steps performed:
1. Loaded IRMOF-13 framework structure
2. Set up Argon as probe molecule for surface area calculation
3. Created simulation input with surface area calculation parameters:
   - SimulationType: MonteCarlo
   - ComputeSurfaceArea: enabled
   - SurfaceAreaSamplingPointsPerSphere: 1000 points
   - SurfaceAreaProbeDistance: Minimum (uses 2^(1/6)σ ≈ 1.12246σ)
   - SurfaceAreaProbeAtom: Argon
   - Framework: 2x2x2 unit cells at 298K, 1e5 Pa
   - Component: Argon with SurfaceAreaProbability 1.0, CreateNumberOfMolecules 0
4. Executed RASPA simulation
5. Parsed output for surface area results

Method: Geometric surface area calculated by 'rolling an atom over the surface' - generating points on sphere around each framework atom and measuring overlap with other framework atoms.

Results: Surface area values reported in [m²/cm³] and [m²/g] units.

Note: This is a pure geometric calculation based on framework structure only, no prerequisites required.