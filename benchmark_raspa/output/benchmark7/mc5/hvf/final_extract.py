#!/usr/bin/env python3

# Final attempt to extract helium void fraction
file_path = 'simulation_1/Output/System_0/output_framework_2.2.1_298.000000_0.data'

try:
    # Read the entire file
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"Successfully read file. Size: {len(content)} characters")
    
    # Split into lines for analysis
    lines = content.split('\n')
    print(f"Total lines: {len(lines)}")
    
    # Search for helium void fraction in various formats
    search_terms = [
        'Rosenbluth weight',
        'rosenbluth weight', 
        'Average Rosenbluth weight',
        'average rosenbluth weight',
        'Helium void fraction',
        'helium void fraction',
        'Void fraction',
        'void fraction',
        'Widom insertion',
        'widom insertion'
    ]
    
    results_found = []
    
    for i, line in enumerate(lines):
        for term in search_terms:
            if term in line:
                results_found.append((i, term, line.strip()))
    
    if results_found:
        print("\n=== HELIUM VOID FRACTION RESULTS ===\n")
        for line_num, term, line_content in results_found:
            print(f"Found '{term}' at line {line_num}:")
            print(f"  {line_content}")
            
            # Print surrounding context
            for j in range(max(0, line_num-2), min(len(lines), line_num+3)):
                if j != line_num and lines[j].strip():
                    print(f"  Context {j}: {lines[j].strip()}")
            print("\n---\n")
    else:
        print("No direct helium void fraction terms found.")
        print("\nSearching for numerical results in the last part of the file...")
        
        # Look at the end of the file for results
        print("\n=== FINAL RESULTS SECTION ===\n")
        for i in range(max(0, len(lines)-50), len(lines)):
            if lines[i].strip() and any(char.isdigit() for char in lines[i]):
                print(f"{i}: {lines[i].strip()}")
    
    # Also search for any mention of Component 0 (helium) results
    print("\n=== COMPONENT 0 (HELIUM) RESULTS ===\n")
    for i, line in enumerate(lines):
        if 'Component 0' in line or 'component 0' in line:
            print(f"Line {i}: {line.strip()}")
            # Print next few lines for context
            for j in range(i+1, min(len(lines), i+5)):
                if lines[j].strip():
                    print(f"  {j}: {lines[j].strip()}")
            print("---")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nScript completed.")