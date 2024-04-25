#!/bin/bash

PYTHON=$(which python)
SCRIPTS_DIR="scripts"

alias l='ls -lhtr'

if [ ! -d "$SCRIPTS_DIR" ]; then
    echo "Error: $SCRIPTS_DIR directory not found."
    exit 1
fi

for file in "$SCRIPTS_DIR"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        extension="${filename##*.}"
        
        if [ "$extension" == "sh" ]; then
            script_name="${filename%.*}"
            alias $script_name="./$SCRIPTS_DIR/$filename"
            echo "Alias for $filename created."
        
        elif [ "$extension" == "py" ]; then
            # Create an alias using pythonpath
            script_name="${filename%.*}"
            alias $script_name="$PYTHON $SCRIPTS_DIR/$filename"
            echo "Alias for $filename created."
        fi
    fi
done
