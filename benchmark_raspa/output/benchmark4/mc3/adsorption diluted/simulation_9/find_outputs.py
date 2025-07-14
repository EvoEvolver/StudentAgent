import os
import glob

# Search for any output files
for root, dirs, files in os.walk('.'):
    for file in files:
        if 'output' in file.lower() or file.endswith('.data'):
            print(f'Found: {os.path.join(root, file)}')

# Also check for any .data files
data_files = glob.glob('**/*.data', recursive=True)
print('Data files found:', data_files)

# Check current directory contents
print('Current directory contents:')
for item in os.listdir('.'):
    print(f'  {item}')
