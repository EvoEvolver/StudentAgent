HELIUM VOID FRACTION CALCULATION FOR IRMOF-13
==============================================

This simulation setup calculates the helium void fraction of IRMOF-13 framework.

STEPS COMPLETED:
1. Loaded IRMOF-13 framework as framework.cif with unit cells [2, 2, 1]
2. Generated helium molecule definition files (helium.def, force_field.def, etc.)
3. Created simulation.input with Monte Carlo simulation using Widom insertion method

SIMULATION PARAMETERS:
- SimulationType: MonteCarlo
- NumberOfCycles: 1000 (reduced from typical 10,000+ for faster execution)
- NumberOfInitializationCycles: 500
- WidomProbability: 1.0 (essential for void fraction calculation)
- Component: Helium with CreateNumberOfMolecules: 0
- Temperature: 298.0 K
- Pressure: 1e5 Pa

IMPORTANT NOTES:
- This is a PREREQUISITE simulation that must be completed before any main framework simulations
- The helium void fraction result will be found as "Average Widom Rosenbluth factor" in the output
- This value must be used as HeliumVoidFraction parameter in subsequent RASPA simulations with this framework
- Simulation steps reduced to 1/10 of typical values for faster execution as requested

TO EXECUTE:
Run 'execute raspa' command to start the simulation.

OUTPUT ANALYSIS:
Use 'output_parser' tool to extract the helium void fraction value from the results.