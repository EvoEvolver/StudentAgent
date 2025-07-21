#!/usr/bin/env python3

# Extract complete helium void fraction from RASPA output
file_path = 'simulation_1/Output/System_0/output_framework_2.2.1_298.000000_0.data'

try:
    with open(file_path, 'r') as f:
        all_lines = f.readlines()
    
    print(f"Successfully read {len(all_lines)} lines from output file")
    
    # Search for helium void fraction
    void_fraction_value = None
    
    for i, line in enumerate(all_lines):
        line = line.strip()
        
        # Look for Rosenbluth weight (this is the helium void fraction)
        if 'Rosenbluth' in line or 'rosenbluth' in line:
            print(f"Line {i}: {line}")
            
            # Extract numerical value
            import re
            numbers = re.findall(r'\d+\.\d+(?:[eE][+-]?\d+)?', line)
            if numbers:
                void_fraction_value = numbers[0]
                print(f"Found helium void fraction: {void_fraction_value}")
        
        # Also look for explicit void fraction mentions
        if 'void fraction' in line.lower():
            print(f"Line {i}: {line}")
            
            import re
            numbers = re.findall(r'\d+\.\d+(?:[eE][+-]?\d+)?', line)
            if numbers:
                void_fraction_value = numbers[0]
                print(f"Found helium void fraction: {void_fraction_value}")
    
    # If not found, print the entire file to see what's there
    if void_fraction_value is None:
        print("\nHelium void fraction not found with keywords. Printing entire file:")
        print("=" * 80)
        for i, line in enumerate(all_lines):
            print(f"{i:4d}: {line.rstrip()}")
        print("=" * 80)
    else:
        print(f"\n*** RESULT: Helium void fraction for IRMOF-13 = {void_fraction_value} ***")
        
except Exception as e:
    print(f"Error reading file: {e}")
    import traceback
    traceback.print_exc()
