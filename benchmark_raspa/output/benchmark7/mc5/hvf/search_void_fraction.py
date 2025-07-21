#!/usr/bin/env python3
import re

# Read the complete output file
try:
    with open('simulation_1/Output/System_0/output_framework_2.2.1_298.000000_0.data', 'r') as f:
        content = f.read()
    
    print("File length:", len(content))
    print("\n=== Searching for Rosenbluth weight ===\n")
    
    # Search for Rosenbluth weight patterns
    rosenbluth_patterns = [
        r'.*[Rr]osenbluth.*',
        r'.*[Vv]oid.*[Ff]raction.*',
        r'.*[Aa]verage.*[Ww]eight.*',
        r'.*[Ii]nsertion.*[Pp]robability.*',
        r'.*[Hh]elium.*'
    ]
    
    lines = content.split('\n')
    found_results = False
    
    for pattern in rosenbluth_patterns:
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                print(f"Pattern '{pattern}' found at line {i}:")
                print(f"  {line}")
                # Print context
                for j in range(max(0, i-2), min(len(lines), i+3)):
                    if j != i:
                        print(f"    {j}: {lines[j]}")
                print("---")
                found_results = True
    
    if not found_results:
        print("No Rosenbluth weight patterns found. Searching for numerical results...")
        
        # Look for sections with numerical data
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['average', 'final', 'result', 'summary']):
                if any(char.isdigit() for char in line):
                    print(f"Numerical line {i}: {line}")
                    for j in range(max(0, i-1), min(len(lines), i+2)):
                        if j != i:
                            print(f"  {j}: {lines[j]}")
                    print("---")
    
    # Also search the end of the file for final results
    print("\n=== Last 50 lines of output ===\n")
    for i, line in enumerate(lines[-50:], len(lines)-50):
        print(f"{i}: {line}")
        
except Exception as e:
    print(f"Error reading file: {e}")
