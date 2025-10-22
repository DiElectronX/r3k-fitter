#!/bin/bash

# Check if a path argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path>"
    exit 1
fi

# Assign the path variable
TARGET_PATH="$1"
CFG_FILE="../new_trigger_cfg.yml"

# Validate that the provided path exists and is a directory
if [ ! -d "$TARGET_PATH" ]; then
    echo "Error: $TARGET_PATH is not a valid directory."
    exit 1
fi

# Define categories with their associated search patterns
declare -A CATEGORY_PATTERNS
CATEGORY_PATTERNS["jpsi"]="*_jpsi_* *_jpsi.*"
CATEGORY_PATTERNS["psi2s"]="*_psi2s_* *_psi2s.*"
CATEGORY_PATTERNS["lowq2"]="*measurement_rare* *k*star_kaon* *k*star_pion*"

# Iterate over categories
for category in "${!CATEGORY_PATTERNS[@]}"; do
    patterns=${CATEGORY_PATTERNS[$category]}
    echo "Processing category: $category with patterns: $patterns"

    # Iterate over each pattern in the category
    for pattern in $patterns; do
        echo "  Searching with pattern: $pattern"
        
        # Find files that match the current pattern
        find "$TARGET_PATH" -type f -name "$pattern" | while read -r file; do
            echo "    Processing file: $file"

            # Run the Python script with the required arguments
            python3 get_weighted_sum.py -c "$CFG_FILE" -i "$file" -m "$category"

            # Check for errors
            if [ $? -ne 0 ]; then
                echo "Error: Python script failed for file $file with category $category."
            fi
        done
    done
done

echo "Processing complete."

