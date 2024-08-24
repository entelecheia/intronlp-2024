#!/bin/bash

# Load the configuration file
source book/_config.env

# Format the languages as a JavaScript array
formatted_languages=$(printf "'%s'," "${LANGUAGES[@]}")
formatted_languages="[${formatted_languages%,}]"

# Define the languages and their respective build directories
LANGUAGE_SWITCHER_FILE="book/_addons/language_switcher.js"
LANGUAGE_SELECTOR_CSS_FILE="book/_addons/language_selector.css"

# Main execution
echo "Starting pre-processing..."

# Copy addon files to the build directory
for i in "${!LANGUAGES[@]}"; do
    lang=${LANGUAGES[$i]}
    echo "Processing $lang..."
    dest_dir="book/${lang}"
    static_dest_dir="${dest_dir}/_static"
    mkdir -p "${static_dest_dir}"
    echo "Copying CSS files..."
    cp -f "$LANGUAGE_SELECTOR_CSS_FILE" "$static_dest_dir/"
    # Copy language_switcher.js to the _static directory
    echo "Copying language_switcher.js..."
    cp -f "$LANGUAGE_SWITCHER_FILE" "$static_dest_dir/"
    # Replace the supported languages placeholder in language_switcher.js
    lang_switcher_file="${static_dest_dir}/language_switcher.js"
    sed -i.bak "s/__SUPPORTED_LANGUAGES__/$formatted_languages/" "$lang_switcher_file"
    # Remove backup file
    rm "$lang_switcher_file.bak"
done

echo "Pre-processing complete!"
