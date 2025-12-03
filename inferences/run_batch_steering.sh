#!/bin/bash

# Batch steering test with SMALL strengths (0.5-2.0) to avoid repetition

echo "=========================================="
echo "Batch Feature Steering Test"
echo "Using small strengths: 0.5, 1.0, 2.0"
echo "=========================================="
echo ""

python inferences/batch_steering_test.py \
    --num-samples 16 \
    --max-new-tokens 128 \
    --temperature 0.7

echo ""
echo "=========================================="
echo "Test completed!"
echo "Check results/batch_steering/ for outputs"
echo "=========================================="
