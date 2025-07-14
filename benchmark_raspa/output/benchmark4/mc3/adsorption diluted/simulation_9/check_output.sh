#!/bin/bash
find . -name "*.data" -type f
find . -name "output*" -type f
ls -la Output/System_0/ 2>/dev/null || echo "Output directory not found"
