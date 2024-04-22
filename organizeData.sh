#!/bin/bash

# Define the zip file and extraction path
ZIP_FILE="SUR_projekt2023-2024.zip"
EXTRACT_PATH="./extracted"

echo "Extracting data from $ZIP_FILE..."
# Unzip the file into the extract path
unzip -q $ZIP_FILE -d $EXTRACT_PATH

# Create the new directory structure if it does not exist
mkdir -p data/dev data/train

# Remove old directories if they exist and move the new ones
rm -rf data/dev/non_target_dev data/dev/target_dev data/train/non_target_train data/train/target_train

mv $EXTRACT_PATH/non_target_dev data/dev/
mv $EXTRACT_PATH/target_dev data/dev/
mv $EXTRACT_PATH/non_target_train data/train/
mv $EXTRACT_PATH/target_train data/train/

echo "Data organized in ./data directory."

# Remove the temporary extracted directory
rm -rf $EXTRACT_PATH
