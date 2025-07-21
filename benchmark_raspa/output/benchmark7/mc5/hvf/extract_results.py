#!/usr/bin/env python3

# Extract helium void fraction from RASPA output
file_path = 'simulation_1/Output/System_0/output_framework_2.2.1_298.000000_0.data'

try:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    print("\n=== Searching for Rosenbluth weight and void fraction ===\n")
    
    # Search for key patterns
    keywords = ['rosenbluth', 'void', 'fraction', 'weight', 'insertion', 'helium', 'average']
    
    found_lines = []
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in keywords):
            found_lines.append((i, line.strip()))
    
    if found_lines:
        print("Found relevant lines:")
        for line_num, line_content in found_lines:
            print(f"Line {line_num}: {line_content}")
    else:
        print("No keyword matches found.")
    
    print("\n=== Last 100 lines of output ===\n")
    start_line = max(0, len(lines) - 100)
    for i in range(start_line, len(lines)):
        print(f"{i}: {lines[i].rstrip()}")
        
except Exception as e:
    print(f"Error: {e}")
