bash
#!/bin/bash

# Parent directory containing the keyframes_Videos_LXX folders
PARENT_DIR="/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f41/data_aichallenge2025"

# Path to the Python script
PYTHON_SCRIPT="Internvideo_feats.py"

# Group size and stride for the commands
GROUP_SIZE=3
STRIDE=1

# Check if the Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script $PYTHON_SCRIPT not found in the current directory."
    exit 1
fi

# Find all folders matching the pattern keyframes_Videos_LXX
for FOLDER in "$PARENT_DIR"/keyframes_Videos_L*; do
    # Extract the folder name (e.g., keyframes_Videos_L21)
    FOLDER_NAME=$(basename "$FOLDER")
    
    # Check if the folder exists and is a directory
    if [[ -d "$FOLDER" ]]; then
        echo "Processing folder: $FOLDER_NAME"
        
        # Define input and output paths
        INPUT_KEYFRAMES="$FOLDER/keyframes"
        OUTPUT_NPY="$FOLDER/npy"
        OUTPUT_BIN="$FOLDER"
        
        # Create output directories if they don't exist
        mkdir -p "$OUTPUT_NPY"
        mkdir -p "$OUTPUT_BIN"
        
        # Run img2npy command
        echo "Running img2npy for $FOLDER_NAME..."
        python3 "$PYTHON_SCRIPT" --img2npy -i "$INPUT_KEYFRAMES" -o "$OUTPUT_NPY" --group_size "$GROUP_SIZE" --stride "$STRIDE"
        
        # Check if img2npy command was successful
        if [[ $? -eq 0 ]]; then
            echo "img2npy completed successfully for $FOLDER_NAME"
        else
            echo "Error: img2npy failed for $FOLDER_NAME"
            continue
        fi
        
        # Run npy2bin command
        echo "Running npy2bin for $FOLDER_NAME..."
        python3 "$PYTHON_SCRIPT" --npy2bin -i "$OUTPUT_NPY" -o "$OUTPUT_BIN" --group_size "$GROUP_SIZE" --stride "$STRIDE"
        
        # Check if npy2bin command was successful
        if [[ $? -eq 0 ]]; then
            echo "npy2bin completed successfully for $FOLDER_NAME"
        else
            echo "Error: npy2bin failed for $FOLDER_NAME"
        fi
    else
        echo "Warning: $FOLDER is not a valid directory, skipping..."
    fi
done

echo "All folders processed."
