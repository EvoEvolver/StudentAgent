#!/usr/bin/env python3

# Read the entire output file
with open('simulation_1/Output/System_0/output_framework_2.2.1_300.000000_100000.data', 'r') as f:
    lines = f.readlines()

print("Searching for enthalpy-related information...")
print("="*60)

# Search for lines containing enthalpy, energy, or heat
search_terms = ['enthalpy', 'Enthalpy', 'ENTHALPY', 'heat', 'Heat', 'HEAT', 'adsorption', 'Adsorption']

found_lines = []
for i, line in enumerate(lines):
    for term in search_terms:
        if term in line:
            found_lines.append((i+1, line.strip()))
            break

if found_lines:
    print("Found relevant lines:")
    for line_num, line in found_lines:
        print(f"Line {line_num}: {line}")
else:
    print("No enthalpy-related lines found.")

# Also search for the end of the file to see if there's more data
print("\nLast 50 lines of the file:")
print("-"*40)
for i, line in enumerate(lines[-50:], len(lines)-49):
    print(f"Line {i}: {line.strip()}")

print("\nSearch completed.")
