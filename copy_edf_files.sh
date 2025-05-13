#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="/Users/wachiii/Workschii/brain-asd/EDF ECEO NC"
DEST_DIR="/Users/wachiii/Workschii/brain-asd/adult_edf_hc"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Find and copy all .edf files from subfolders to the destination directory
find "$SOURCE_DIR" -type f -name "*.edf" -exec cp {} "$DEST_DIR" \;

echo "All .edf files have been copied to $DEST_DIR"