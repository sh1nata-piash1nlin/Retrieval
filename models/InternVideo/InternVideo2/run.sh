#!/bin/bash

# Parent directory containing all keyframes_Videos_L* folders
PARENT_DIR="/workspace/data_aichallenge2025"

# Loop over a specific numeric range (L20 to L30)
for i in $(seq 20 30); do
    folder="$PARENT_DIR/keyframes_Videos_L${i}"
    if [ -d "$folder" ]; then
        echo "Processing folder: $folder"

        python -m multi_modality.Internvideo_feats --img2npy \
            -i "$folder/keyframes" \
            -o "$folder/npy" \
            --group_size 3 --stride 1
    else
        echo "Folder not found: $folder, skipping..."
    fi
done
