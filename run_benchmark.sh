#!/bin/bash

# Benchmark command
# Other Available models: gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o, gpt-4o-mini, together_ai/meta-llama/Llama-3-70b-chat-hf, etc.

# sotopia benchmark \
#   --models gpt-5-nano \
#   --partner-model gpt-5-nano \
#   --evaluator-model gpt-5-mini \
#   --push-to-db \
#   --output-to-jsonl \
#   --save-dir ./results \
#   --print-logs 
#   # --only-show-performance

sotopia benchmark \
  --models custom/qwen/qwen3-1.7b@http://127.0.0.1:1234/v1 \
  --partner-model custom/qwen/qwen3-1.7b@http://127.0.0.1:1234/v1 \
  --evaluator-model custom/openai/gpt-oss-20b@http://127.0.0.1:1234/v1 \
  --push-to-db \
  --output-to-jsonl \
  --save-dir ./results \
  --print-logs 
  # --only-show-performance
