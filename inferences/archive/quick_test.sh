#!/bin/bash
# Quick test script for feature steering

echo "Quick Feature Steering Test"
echo "============================"
echo ""

# Simple test prompt
TEXT="Hello, I would like to discuss our collaboration. How can we work together effectively?"

echo "Testing feature 325 (relationship-related, correlation: 0.5276)"
echo "This will compare baseline vs enhanced relationship feature"
echo ""

python inferences/qwen_sae_feature_steering.py \
    --feature-idx 325 \
    --steering-strength 5.0 \
    --text "$TEXT" \
    --max-new-tokens 128 \
    --temperature 0.7

echo ""
echo "Test completed! Check the output above to see the difference."
