#!/bin/bash

# Check if a path argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path>"
    exit 1
fi

# Assign the path variable
TARGET_PATH="$1"

# Validate that the provided path exists and is a directory
if [ ! -d "$TARGET_PATH" ]; then
    echo "Error: $TARGET_PATH is not a valid directory."
    exit 1
fi

# Define search patterns and their corresponding categories
declare -A SEARCH_PATTERNS
SEARCH_PATTERNS["*_jpsi_*"]="jpsi"
SEARCH_PATTERNS["*_psi2s_*"]="psi2s"
SEARCH_PATTERNS["*measurement_rare*"]="lowq2"
SEARCH_PATTERNS["*k*star_kaon*"]="lowq2"
SEARCH_PATTERNS["*k*star_pion*"]="lowq2"

# Iterate over search patterns
for pattern in "${!SEARCH_PATTERNS[@]}"; do
    category="${SEARCH_PATTERNS[$pattern]}"
    echo "Processing files matching pattern: $pattern with category: $category"
    
    # Find files that match the current pattern
    find "$TARGET_PATH" -type f -name "$pattern" | while read -r file; do
        echo "Processing file: $file"
        
        # Run the Python script with the required arguments
        python3 get_weighted_sum.py -c ../fit_cfg_12_9_24.yml -i "$file" -m "$category"
        
        # Check for errors (optional)
        if [ $? -ne 0 ]; then
            echo "Error: Python script failed for file $file with category $category."
        fi
    done
done

echo "Processing complete."

