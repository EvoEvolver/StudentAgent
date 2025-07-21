#!/usr/bin/env python3

# Get helium void fraction from RASPA output
file_path = 'simulation_1/Output/System_0/output_framework_2.2.1_298.000000_0.data'

try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    
    # Search for helium void fraction patterns
    lines = content.split('\n')
    print(f"Total lines: {len(lines)}")
    
    # Look for the actual helium void fraction result
    void_fraction = None
    
    for i, line in enumerate(lines):
        # Check for Rosenbluth weight patterns
        if 'Rosenbluth' in line or 'rosenbluth' in line:
            print(f"Found Rosenbluth at line {i}: {line.strip()}")
            
            # Try to extract numerical value
            import re
            numbers = re.findall(r'\d+\.\d+', line)
            if numbers:
                void_fraction = float(numbers[0])
                print(f"Extracted void fraction: {void_fraction}")
        
        # Also check for explicit void fraction mentions
        if 'void fraction' in line.lower() or 'helium void fraction' in line.lower():
            print(f"Found void fraction at line {i}: {line.strip()}")
            
            import re
            numbers = re.findall(r'\d+\.\d+', line)
            if numbers:
                void_fraction = float(numbers[0])
                print(f"Extracted void fraction: {void_fraction}")
    
    if void_fraction is None:
        print("\nNo explicit void fraction found. Printing last 100 lines to check for results:")
        for i in range(max(0, len(lines)-100), len(lines)):
            if lines[i].strip():
                print(f"{i}: {lines[i].strip()}")
    else:
        print(f"\n=== HELIUM VOID FRACTION FOR IRMOF-13 ===\n")
        print(f"Helium void fraction: {void_fraction}")
        
except Exception as e:
    print(f"Error: {e}")
