#!/bin/bash

# Define the target directory
TARGET_DIR="/Users/wachiii/Workschii/brain-asd/data_adults_eyeopen"

# Recursively find and delete files ending with "_EO.fif"
find "$TARGET_DIR" -type f -name "*_EC.fif" -exec rm -v {} \;

echo "All files ending with '_EO.fif' have been deleted."