#!/usr/bin/env bash
set -e

# Usage:
#   ./run_sotopia_eval_steered.sh <feature_idx> <strength> [base_model] [port]
#
# This script is designed for the setup where:
#   - The Qwen SAE server runs on a remote GPU machine.
#   - You run Sotopia eval locally and connect via an HTTP endpoint
#     that looks like /v1/chat/completions.
#   - Different experiments are identified by different SAE feature indices
#     (and optionally strengths), without restarting the remote server.
#
# It encodes the steering config into the MODEL_NAME string so that
# the remote server can parse it from req.model and apply the correct
# feature/strength per request.
#
# The server-side convention (in qwen_sae_server_residual.py) is:
#   model name contains:  __feat{idx}_str{strength}
# Example MODEL_NAME:
#   Qwen/Qwen2.5-7B-Instruct__feat123_str5.0

FEATURE_IDX=$1
STRENGTH=$2
BASE_MODEL="${3:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${4:-8000}"

if [ -z "$FEATURE_IDX" ] || [ -z "$STRENGTH" ]; then
    echo "Usage: $0 <feature_idx> <strength> [base_model] [port]"
    exit 1
fi

# Encode steering config into the model name; this will be visible on the
# server side as req.model and parsed by parse_feature_from_model_name.
MODEL_NAME="${BASE_MODEL}__feat${FEATURE_IDX}_str${STRENGTH}"

echo "Running Sotopia eval with:"
echo "  BASE_MODEL   = ${BASE_MODEL}"
echo "  FEATURE_IDX  = ${FEATURE_IDX}"
echo "  STRENGTH     = ${STRENGTH}"
echo "  MODEL_NAME   = ${MODEL_NAME}"
echo "  PORT         = ${PORT}"

bash run_sotopia_eval.sh "${MODEL_NAME}" "${PORT}"

