#!/bin/bash
# Search for Rosenbluth weight information
grep -i -A5 -B5 'rosenbluth\|widom\|insertion' simulation_1/Output/System_0/output_Box_1.1.1_298.000000_0.data
