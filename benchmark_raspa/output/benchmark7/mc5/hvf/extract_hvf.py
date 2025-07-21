#!/usr/bin/env python3

# Extract helium void fraction from RASPA output
file_path = 'simulation_1/Output/System_0/output_framework_2.2.1_298.000000_0.data'

try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"File read successfully. Length: {len(content)} characters")
    
    # Search for helium void fraction patterns
    lines = content.split('\n')
    
    # Look for the specific output format from RASPA
    void_fraction_found = False
    
    for i, line in enumerate(lines):
        # Check for various patterns that might contain the void fraction
        if any(keyword in line.lower() for keyword in ['rosenbluth', 'void', 'fraction', 'helium']):
            print(f"Found relevant line {i}: {line.strip()}")
            void_fraction_found = True
            
            # Print context
            for j in range(max(0, i-2), min(len(lines), i+3)):
                if j != i:
                    print(f"  Context {j}: {lines[j].strip()}")
            print("---")
    
    if not void_fraction_found:
        print("No direct void fraction patterns found. Checking end of file...")
        
        # Print last 100 lines to see final results
        print("\n=== LAST 100 LINES OF OUTPUT ===\n")
        start_idx = max(0, len(lines) - 100)
        for i in range(start_idx, len(lines)):
            if lines[i].strip():  # Only non-empty lines
                print(f"{i}: {lines[i].strip()}")
    
    # Also search for numerical patterns that might be the result
    import re
    print("\n=== SEARCHING FOR NUMERICAL RESULTS ===\n")
    for i, line in enumerate(lines):
        # Look for lines with floating point numbers
        if re.search(r'\d+\.\d+', line):
            line_lower = line.lower()
            if any(word in line_lower for word in ['average', 'final', 'result', 'weight', 'probability']):
                print(f"Numerical result line {i}: {line.strip()}")
                
except Exception as e:
    print(f"Error reading file: {e}")
    import traceback
    traceback.print_exc()
