# Calculation of adsorption enthalpy for n-heptane on IRMOF-13
# From RASPA simulation results at infinite dilution (298 K)

# Data from simulation_3 output:
host_guest_energy = -3471.98468  # K (Average Host-Adsorbate energy)
temperature = 298.0  # K
R_gas_constant = 8.314  # J/(mol·K)
boltzmann_constant = 0.8314464919  # RASPA internal units

# For infinite dilution and rigid framework:
# ΔH = <U_hg> - RT
# Converting from K to kJ/mol: multiply by R_gas_constant/1000

# Method 1: Direct conversion from K to kJ/mol
adsorption_enthalpy_kJ_mol = (host_guest_energy - temperature) * R_gas_constant / 1000

# Method 2: Using RASPA's internal conversion
# Energy to Kelvin factor from RASPA output: 1.2027242847
energy_to_kelvin = 1.2027242847
adsorption_enthalpy_kJ_mol_alt = (host_guest_energy - temperature) * boltzmann_constant / 1000

print(f"Adsorption enthalpy calculation for n-heptane on IRMOF-13:")
print(f"Host-Guest interaction energy: {host_guest_energy:.2f} K")
print(f"Temperature: {temperature} K")
print(f"")
print(f"Method 1 - Direct conversion:")
print(f"ΔH_ads = ({host_guest_energy:.2f} - {temperature}) × {R_gas_constant}/1000")
print(f"ΔH_ads = {adsorption_enthalpy_kJ_mol:.2f} kJ/mol")
print(f"")
print(f"Method 2 - RASPA internal units:")
print(f"ΔH_ads = {adsorption_enthalpy_kJ_mol_alt:.2f} kJ/mol")
print(f"")
print(f"Final result: ΔH_ads ≈ {adsorption_enthalpy_kJ_mol:.1f} kJ/mol")
