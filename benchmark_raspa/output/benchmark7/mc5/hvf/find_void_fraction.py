#!/usr/bin/env python3

# Find helium void fraction in RASPA output
file_path = 'simulation_1/Output/System_0/output_framework_2.2.1_298.000000_0.data'

try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    
    # Split into lines
    lines = content.split('\n')
    print(f"Total lines: {len(lines)}")
    
    # Look for specific patterns related to Widom insertion and Rosenbluth weight
    patterns_to_search = [
        'Rosenbluth',
        'rosenbluth', 
        'Widom',
        'widom',
        'void fraction',
        'Void fraction',
        'VOID FRACTION',
        'Average Rosenbluth weight',
        'insertion probability',
        'Insertion probability'
    ]
    
    found_results = []
    
    for i, line in enumerate(lines):
        for pattern in patterns_to_search:
            if pattern in line:
                found_results.append((i, pattern, line.strip()))
    
    if found_results:
        print("\n=== Found relevant patterns ===\n")
        for line_num, pattern, line_content in found_results:
            print(f"Line {line_num} (pattern: {pattern}): {line_content}")
            # Print context lines
            for j in range(max(0, line_num-2), min(len(lines), line_num+3)):
                if j != line_num:
                    print(f"  {j}: {lines[j].strip()}")
            print("---")
    else:
        print("No specific patterns found. Searching for numerical results...")
        
        # Search for lines with numbers that might be the result
        import re
        for i, line in enumerate(lines):
            if re.search(r'\d+\.\d+', line) and ('average' in line.lower() or 'final' in line.lower()):
                print(f"Line {i}: {line.strip()}")
    
    # Print the last 50 lines to see final results
    print("\n=== Last 50 lines ===\n")
    for i in range(max(0, len(lines)-50), len(lines)):
        if lines[i].strip():  # Only print non-empty lines
            print(f"{i}: {lines[i].strip()}")
            
except Exception as e:
    print(f"Error reading file: {e}")
