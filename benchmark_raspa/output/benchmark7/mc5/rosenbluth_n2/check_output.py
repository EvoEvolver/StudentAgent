#!/usr/bin/env python3

# Check the RASPA output file for Rosenbluth weight information
file_path = 'simulation_1/Output/System_0/output_Box_1.1.1_298.000000_0.data'

try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    lines = content.split('\n')
    print(f"Number of lines: {len(lines)}")
    
    # Search for key patterns
    search_terms = [
        'Rosenbluth', 'rosenbluth', 'ROSENBLUTH',
        'Widom', 'widom', 'WIDOM',
        'insertion', 'Insertion', 'INSERTION',
        'Average', 'AVERAGE',
        'Component 0', 'methane', 'METHANE',
        'Results', 'RESULTS',
        'Final', 'FINAL',
        'Summary', 'SUMMARY'
    ]
    
    found_matches = []
    for i, line in enumerate(lines):
        for term in search_terms:
            if term in line:
                found_matches.append((i+1, term, line.strip()))
    
    if found_matches:
        print("\nFound relevant lines:")
        for line_num, term, line_content in found_matches:
            print(f"Line {line_num} ('{term}'): {line_content}")
            
            # Show context around important matches
            if any(important in term.lower() for important in ['rosenbluth', 'widom', 'results', 'final']):
                print("  Context:")
                start = max(0, line_num-4)
                end = min(len(lines), line_num+3)
                for j in range(start, end):
                    marker = ">>> " if j == line_num-1 else "    "
                    print(f"  {marker}{j+1}: {lines[j].strip()}")
                print()
    else:
        print("\nNo matches found. Showing file structure:")
        
        # Show first 20 lines
        print("\nFirst 20 lines:")
        for i in range(min(20, len(lines))):
            print(f"{i+1}: {lines[i]}")
        
        # Show last 100 lines
        print("\nLast 100 lines:")
        for i, line in enumerate(lines[-100:]):
            line_num = len(lines) - 100 + i + 1
            print(f"{line_num}: {line}")
            
except Exception as e:
    print(f"Error reading file: {e}")
