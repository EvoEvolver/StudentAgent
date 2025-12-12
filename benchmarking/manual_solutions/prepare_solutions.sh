#!/bin/bash

# For each task in ../input/tasks_single.json, create a folder single/{task_name} and copy template/solution.ipynb there as solution.ipynb
# Also handle tasks_multi.json for multi/{task_name}

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INPUT_DIR="${SCRIPT_DIR}/../input"
TEMPLATE_DIR="${SCRIPT_DIR}/template"
SINGLE_DIR="${SCRIPT_DIR}/single"
MULTI_DIR="${SCRIPT_DIR}/multi"

# Function to process tasks from a JSON file
process_tasks() {
    local json_file="$1"
    local output_dir="$2"
    
    if [ ! -f "$json_file" ]; then
        echo "Error: $json_file not found"
        return 1
    fi
    
    echo "Processing $json_file..."
    
    # Extract task names (keys) from the JSON file using Python
    task_names=$(python3 -c "
import json
import sys

with open('$json_file', 'r') as f:
    data = json.load(f)
    for task_name in data.keys():
        print(task_name)
")
    
    # Create a folder for each task and copy template files
    for task_name in $task_names; do
        task_dir="${output_dir}/${task_name}"
        
        # Create task directory if it doesn't exist
        if [ ! -d "$task_dir" ]; then
            echo "  Creating directory: $task_dir"
            mkdir -p "$task_dir"
        else
            echo "  Directory already exists: $task_dir"
        fi
        
        # Copy solution.ipynb from template
        if [ -f "${TEMPLATE_DIR}/solution.ipynb" ]; then
            cp "${TEMPLATE_DIR}/solution.ipynb" "${task_dir}/solution.ipynb"
            echo "    Copied solution.ipynb to $task_dir"
        else
            echo "    Warning: ${TEMPLATE_DIR}/solution.ipynb not found"
        fi
    done
}

# Create output directories if they don't exist
mkdir -p "$SINGLE_DIR"
mkdir -p "$MULTI_DIR"

# Process single tasks
echo "=== Processing Single Tasks ==="
process_tasks "${INPUT_DIR}/tasks_single.json" "$SINGLE_DIR"

echo ""
echo "=== Processing Multi Tasks ==="
process_tasks "${INPUT_DIR}/tasks_multi.json" "$MULTI_DIR"

echo ""
echo "Done! Solution templates have been prepared."
echo "Single task folders: $SINGLE_DIR"
echo "Multi task folders: $MULTI_DIR"
