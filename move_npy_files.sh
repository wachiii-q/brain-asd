#!/bin/bash

SRC_DIR="/Users/wachiii/Workschii/brain-asd/data/nc"
DEST_DIR="/Users/wachiii/Workschii/brain-asd/data/numpync"

mkdir -p "$DEST_DIR"
mv "$SRC_DIR"/*.npy "$DEST_DIR"
echo "All .npy files have been moved from $SRC_DIR to $DEST_DIR"