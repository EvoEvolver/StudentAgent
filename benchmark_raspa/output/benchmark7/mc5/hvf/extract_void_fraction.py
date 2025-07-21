#!/usr/bin/env python3

# Script to extract helium void fraction from RASPA output
with open('simulation_1/Output/System_0/output_framework_2.2.1_298.000000_0.data', 'r') as f:
    content = f.read()
    
# Look for Rosenbluth weight or void fraction keywords
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'Rosenbluth' in line or 'void' in line.lower() or 'fraction' in line.lower():
        print(f"Line {i}: {line}")
        # Print surrounding lines for context
        for j in range(max(0, i-2), min(len(lines), i+3)):
            if j != i:
                print(f"  {j}: {lines[j]}")
        print("---")

# Also look for average values
for i, line in enumerate(lines):
    if 'average' in line.lower() and ('weight' in line.lower() or 'insertion' in line.lower()):
        print(f"Average Line {i}: {line}")
        for j in range(max(0, i-2), min(len(lines), i+3)):
            if j != i:
                print(f"  {j}: {lines[j]}")
        print("---")
