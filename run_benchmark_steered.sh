#!/bin/bash
set -e

# Usage: ./run_benchmark_steered.sh <feature_idx> <strength> [model_name]

FEATURE_IDX=$1
STRENGTH=$2
MODEL_NAME="${3:-Qwen/Qwen2.5-7B-Instruct}"

if [ -z "$FEATURE_IDX" ] || [ -z "$STRENGTH" ]; then
    echo "Usage: $0 <feature_idx> <strength> [model_name]"
    exit 1
fi

echo "========================================================"
echo "Starting Steered Benchmark"
echo "Feature: $FEATURE_IDX"
echo "Strength: $STRENGTH"
echo "Model: $MODEL_NAME"
echo "========================================================"

# 1. Start the Server in Background
echo "Starting SAE Server..."
python inferences/qwen_sae_server_residual.py \
    --base-model-path "$MODEL_NAME" \
    --steer-feature-idx "$FEATURE_IDX" \
    --steer-strength "$STRENGTH" \
    --port 8000 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Ensure server is killed on exit
trap "echo 'Stopping server...'; kill $SERVER_PID" EXIT

# 2. Wait for Server to be Ready
echo "Waiting for server to be ready..."
max_retries=30
count=0
while ! nc -z localhost 8000; do
    sleep 5
    count=$((count+1))
    if [ $count -ge $max_retries ]; then
        echo "Server failed to start within timeout."
        exit 1
    fi
    echo "Waiting... ($count/$max_retries)"
done
echo "Server is up!"

# 3. Run the Benchmark
# We use the existing run_sotopia_eval.sh but skip the server start part if it has one,
# OR we rely on run_sotopia_eval.sh to assume the server is running if we configure it right.
# Looking at run_sotopia_eval.sh, it runs `examples/experiment_eval.py` which connects to AGENT2_MODEL.
# AGENT2_MODEL is set to custom/${MODEL_NAME}@http://localhost:${PORT}/v1

echo "Running Benchmark..."
# Pass the model name and port to run_sotopia_eval.sh
# Note: run_sotopia_eval.sh might try to start its own things or be configured differently.
# Let's assume it uses the environment variables or args we pass.

bash run_sotopia_eval.sh "$MODEL_NAME" 8000

echo "Benchmark finished."
