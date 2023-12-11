#!/bin/bash

# Initialize an empty array to store the parameters
params=()

# Parse command-line arguments
for arg in "$@"; do
    params+=("$arg")
done

# Call your Python script with the parsed parameters
python train.py "${params[@]}"
