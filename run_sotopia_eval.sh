#!/usr/bin/env bash
set -e

# ====== 配置区域 ======
ENV_IDS=(
"01H7VFHNV13MHN97GAH73E3KM8"
)
export ENV_MODEL="custom/gpt-4o-2024-08-06@https://ai-gateway.andrew.cmu.edu/v1"
export AGENT1_MODEL="custom/gpt-4o-2024-08-06@https://ai-gateway.andrew.cmu.edu/v1"
export AGENT2_MODEL="custom/llama3-2-11b-instruct@https://ai-gateway.andrew.cmu.edu/v1"
export TAG="my_tag"
export PUSH_TO_DB="False"

ENV_IDS_STR=$(printf '"%s",' "${ENV_IDS[@]}")
ENV_IDS_STR="[${ENV_IDS_STR%,}]"

cd sotopia

# ====== 执行命令 ======
python examples/experiment_eval.py \
  --gin_file sotopia_conf/generation_utils_conf/generate.gin \
  --gin_file sotopia_conf/server_conf/server.gin \
  --gin_file sotopia_conf/run_async_server_in_batch.gin \
  --gin.BATCH_SIZE=20 \
  --gin.PUSH_TO_DB=False \
  --gin.PRINT_LOGS=True \
  --gin.VERBOSE=True \
  --gin.ENV_IDS="${ENV_IDS_STR}" \
  "--gin.ENV_MODEL='${ENV_MODEL}'" \
  "--gin.AGENT1_MODEL='${AGENT1_MODEL}'" \
  "--gin.AGENT2_MODEL='${AGENT2_MODEL}'" \
  "--gin.TAG='${TAG}'"
