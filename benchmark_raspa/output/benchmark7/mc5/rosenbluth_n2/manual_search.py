#!/usr/bin/env python3

# Manual search for Rosenbluth weight in RASPA output
file_path = 'simulation_1/Output/System_0/output_Box_1.1.1_298.000000_0.data'

try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    
    # Search for various patterns
    patterns = ['Rosenbluth', 'rosenbluth', 'ROSENBLUTH', 'Widom', 'widom', 'WIDOM', 
                'insertion', 'Insertion', 'INSERTION', 'Average', 'AVERAGE',
                'Component 0', 'methane', 'METHANE']
    
    matches = []
    for i, line in enumerate(lines):
        for pattern in patterns:
            if pattern in line:
                matches.append((i+1, pattern, line.strip()))
    
    if matches:
        print("Found matches:")
        for line_num, pattern, line_content in matches:
            print(f"Line {line_num} ('{pattern}'): {line_content}")
            # Print some context
            start = max(0, line_num-3)
            end = min(len(lines), line_num+2)
            print("Context:")
            for j in range(start, end):
                marker = ">>> " if j == line_num-1 else "    "
                print(f"{marker}{j+1}: {lines[j]}")
            print("---")
    else:
        print("No matches found. Showing last 100 lines:")
        for i, line in enumerate(lines[-100:]):
            print(f"{len(lines)-100+i+1}: {line}")
            
except Exception as e:
    print(f"Error: {e}")
