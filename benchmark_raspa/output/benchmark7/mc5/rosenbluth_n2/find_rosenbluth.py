#!/usr/bin/env python3
import os

# Read the output file
file_path = 'simulation_1/Output/System_0/output_Box_1.1.1_298.000000_0.data'

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Look for Rosenbluth weight information
    found_lines = []
    for i, line in enumerate(lines):
        if any(keyword.lower() in line.lower() for keyword in ['rosenbluth', 'widom', 'insertion', 'average']):
            found_lines.append((i+1, line.strip()))
    
    if found_lines:
        print("Found potential Rosenbluth weight information:")
        for line_num, line_content in found_lines:
            print(f"Line {line_num}: {line_content}")
    else:
        print("No Rosenbluth weight information found with keywords.")
        print("Let's check the last 50 lines of the file:")
        for i, line in enumerate(lines[-50:]):
            print(f"{len(lines)-50+i+1}: {line.strip()}")
else:
    print(f"File {file_path} not found.")
