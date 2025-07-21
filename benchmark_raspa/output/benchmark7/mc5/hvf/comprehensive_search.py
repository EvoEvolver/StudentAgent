#!/usr/bin/env python3

# Comprehensive search for helium void fraction
file_path = 'simulation_1/Output/System_0/output_framework_2.2.1_298.000000_0.data'

try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"File read successfully. Size: {len(content)} characters")
    
    lines = content.split('\n')
    print(f"Total lines: {len(lines)}")
    
    # Print the entire file content to see what's actually there
    print("\n=== COMPLETE FILE CONTENT ===\n")
    for i, line in enumerate(lines):
        if line.strip():  # Only print non-empty lines
            print(f"{i:4d}: {line}")
    
    print("\n=== END OF FILE ===\n")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
