#!/bin/bash

# Example script for running SAE feature steering

# Example prompt (replace with your Sotopia prompt)
TEXT="You are in a social conversation. Please respond politely and build good relationships with others."

# Example 1: Steer feature 325 (relationship-related, positive correlation)
echo "=========================================="
echo "Example 1: Steer feature 325 (enhance)"
echo "=========================================="
python inferences/qwen_sae_feature_steering.py \
    --feature-idx 325 \
    --steering-strength 5.0 \
    --text "$TEXT" \
    --max-new-tokens 256 \
    --output-file "results/steering_feature325_enhance.json"

# Example 2: Steer feature 226 (believability-related, negative correlation)
echo ""
echo "=========================================="
echo "Example 2: Steer feature 226 (suppress)"
echo "=========================================="
python inferences/qwen_sae_feature_steering.py \
    --feature-idx 226 \
    --steering-strength -3.0 \
    --text "$TEXT" \
    --max-new-tokens 256 \
    --output-file "results/steering_feature226_suppress.json"

# Example 3: Test different strengths for the same feature
echo ""
echo "=========================================="
echo "Example 3: Feature 116 with strength 3.0"
echo "=========================================="
python inferences/qwen_sae_feature_steering.py \
    --feature-idx 116 \
    --steering-strength 3.0 \
    --text "$TEXT" \
    --max-new-tokens 256 \
    --output-file "results/steering_feature116_3.json"

echo ""
echo "=========================================="
echo "All examples completed!"
echo "=========================================="