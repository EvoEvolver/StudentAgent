#!/usr/bin/env python3

# Extract the end of the RASPA output file to find results
file_path = 'simulation_1/Output/System_0/output_Box_1.1.1_298.000000_0.data'

try:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    print("\nLast 200 lines of the file:")
    print("=" * 50)
    
    for i, line in enumerate(lines[-200:]):
        line_num = len(lines) - 200 + i + 1
        print(f"{line_num:4d}: {line.rstrip()}")
        
except Exception as e:
    print(f"Error reading file: {e}")
