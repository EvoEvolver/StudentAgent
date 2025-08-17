RASPA Simulation Setup for Ethane Adsorption Enthalpy on IRMOF-13
==================================================================

Objective: Determine the adsorption enthalpy of ethane on IRMOF-13 at infinite dilution

Steps performed:

1. Framework Setup:
   - Loaded IRMOF-13 framework using framework loader
   - Framework saved as framework.cif
   - Unit cells: 2x2x2 (sufficient for 12.8 Å cutoff)

2. Molecule Setup:
   - Loaded helium molecule for void fraction calculation
   - Loaded ethane molecule for adsorption simulation
   - Generated corresponding .def and force field files

3. Simulation Files Created:
   
   a) simulation.input (Helium void fraction calculation):
      - Monte Carlo simulation with 1000 cycles (10% of typical)
      - 32 helium molecules maximum
      - Temperature: 298 K, Pressure: 1 bar
      - Purpose: Calculate accessible volume of IRMOF-13
   
   b) simulation_ethane_adsorption.input (Adsorption enthalpy):
      - Monte Carlo simulation with 1000 cycles (10% of typical)
      - 1 ethane molecule (infinite dilution condition)
      - Widom insertion method enabled for chemical potential
      - Energy histogram computation enabled
      - Temperature: 298 K, Pressure: 1 bar

4. Simulation Procedure:
   - First run helium void fraction simulation
   - Update HeliumVoidFraction value in ethane simulation file
   - Run ethane adsorption simulation
   - Extract adsorption enthalpy from energy histograms

Note: Simulation cycles reduced to 10% of typical values for faster execution as requested.
