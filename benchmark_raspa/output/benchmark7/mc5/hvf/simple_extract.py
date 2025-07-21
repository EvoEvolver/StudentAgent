#!/usr/bin/env python3

# Simple extraction of helium void fraction
file_path = 'simulation_1/Output/System_0/output_framework_2.2.1_298.000000_0.data'

try:
    with open(file_path, 'r') as f:
        all_text = f.read()
    
    # Search for the specific text patterns that indicate helium void fraction
    if 'Rosenbluth' in all_text:
        print("Found 'Rosenbluth' in output!")
        lines = all_text.split('\n')
        for i, line in enumerate(lines):
            if 'Rosenbluth' in line:
                print(f"Line {i}: {line}")
                # Print surrounding context
                for j in range(max(0, i-3), min(len(lines), i+4)):
                    if j != i:
                        print(f"  {j}: {lines[j]}")
                print("\n---\n")
    else:
        print("'Rosenbluth' not found. Searching for other patterns...")
        
        # Look for numerical results at the end
        lines = all_text.split('\n')
        print("\nLast 30 lines of output:")
        for i in range(max(0, len(lines)-30), len(lines)):
            if lines[i].strip():
                print(f"{i}: {lines[i]}")
                
        # Also search for any line containing numbers that might be the void fraction
        import re
        print("\nLines with decimal numbers:")
        for i, line in enumerate(lines[-100:], len(lines)-100):  # Last 100 lines
            if re.search(r'\d+\.\d+', line) and line.strip():
                print(f"{i}: {line.strip()}")
                
except Exception as e:
    print(f"Error: {e}")
