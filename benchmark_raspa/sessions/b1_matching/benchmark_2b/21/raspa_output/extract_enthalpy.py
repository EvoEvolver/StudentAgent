#!/usr/bin/env python3
import re

# Read the output file
with open('simulation_1/Output/System_0/output_framework_2.2.1_300.000000_100000.data', 'r') as f:
    content = f.read()

# Search for enthalpy-related sections
enthalpy_patterns = [
    r'Enthalpy of.*?adsorption.*?([+-]?\d+\.\d+).*?kJ/mol',
    r'Heat of adsorption.*?([+-]?\d+\.\d+).*?kJ/mol',
    r'Adsorption enthalpy.*?([+-]?\d+\.\d+).*?kJ/mol',
    r'Enthalpy.*?([+-]?\d+\.\d+).*?J/mol',
    r'Average.*?enthalpy.*?([+-]?\d+\.\d+)',
    r'Enthalpy.*?([+-]?\d+\.\d+)',
]

print("Searching for enthalpy data in RASPA output...")
print("="*50)

for i, pattern in enumerate(enthalpy_patterns):
    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
    if matches:
        print(f"Pattern {i+1} found: {matches}")

# Also search for any line containing 'enthalpy' or 'Enthalpy'
enthalpy_lines = []
for line_num, line in enumerate(content.split('\n'), 1):
    if 'enthalpy' in line.lower():
        enthalpy_lines.append((line_num, line.strip()))

if enthalpy_lines:
    print("\nLines containing 'enthalpy':")
    for line_num, line in enthalpy_lines:
        print(f"Line {line_num}: {line}")
else:
    print("\nNo lines containing 'enthalpy' found.")

# Search for energy-related data that might contain enthalpy
print("\nSearching for energy-related data...")
energy_patterns = [
    r'Average.*?energy.*?([+-]?\d+\.\d+)',
    r'Total.*?energy.*?([+-]?\d+\.\d+)',
    r'Host/Adsorbate.*?energy.*?([+-]?\d+\.\d+)',
]

for pattern in energy_patterns:
    matches = re.findall(pattern, content, re.IGNORECASE)
    if matches:
        print(f"Energy data found: {matches}")

print("\nScript completed.")
