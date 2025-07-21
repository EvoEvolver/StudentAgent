#!/usr/bin/env python3

# Script to search for Rosenbluth weight information in RASPA output
with open('simulation_1/Output/System_0/output_Box_1.1.1_298.000000_0.data', 'r') as f:
    content = f.read()
    
# Search for relevant keywords
keywords = ['Rosenbluth', 'rosenbluth', 'Widom', 'widom', 'insertion', 'Insertion', 'Average', 'AVERAGE']

lines = content.split('\n')
for i, line in enumerate(lines):
    for keyword in keywords:
        if keyword in line:
            # Print context around the match
            start = max(0, i-2)
            end = min(len(lines), i+3)
            print(f"Found '{keyword}' at line {i+1}:")
            for j in range(start, end):
                print(f"{j+1}: {lines[j]}")
            print("---")
