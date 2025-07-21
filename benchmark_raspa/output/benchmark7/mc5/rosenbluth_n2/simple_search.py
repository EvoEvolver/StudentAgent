#!/usr/bin/env python3

# Simple search for Rosenbluth weight information
file_path = 'simulation_1/Output/System_0/output_Box_1.1.1_298.000000_0.data'

try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Look for key sections
    if 'Rosenbluth' in content:
        print("Found 'Rosenbluth' in the file!")
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'Rosenbluth' in line:
                print(f"Line {i+1}: {line}")
                # Print context
                for j in range(max(0, i-3), min(len(lines), i+4)):
                    print(f"  {j+1}: {lines[j]}")
                print("---")
    
    elif 'Widom' in content:
        print("Found 'Widom' in the file!")
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'Widom' in line:
                print(f"Line {i+1}: {line}")
                # Print context
                for j in range(max(0, i-3), min(len(lines), i+4)):
                    print(f"  {j+1}: {lines[j]}")
                print("---")
    
    else:
        print("No 'Rosenbluth' or 'Widom' found. Checking file size and structure...")
        lines = content.split('\n')
        print(f"File has {len(lines)} lines")
        print(f"File size: {len(content)} characters")
        
        # Look for any mention of methane or component results
        methane_lines = []
        for i, line in enumerate(lines):
            if 'methane' in line.lower() or 'component' in line.lower():
                methane_lines.append((i+1, line))
        
        if methane_lines:
            print("\nFound methane/component related lines:")
            for line_num, line in methane_lines:
                print(f"Line {line_num}: {line}")
        
        # Show the last 50 lines
        print("\nLast 50 lines of the file:")
        for i, line in enumerate(lines[-50:]):
            print(f"{len(lines)-50+i+1}: {line}")
            
except Exception as e:
    print(f"Error: {e}")
