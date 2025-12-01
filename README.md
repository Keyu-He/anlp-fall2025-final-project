# anlp-fall2025-final-project

## Run SAE server on Babel

```bash
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
git clone https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct
```

```bash
cd ~/anlp-fall2025-final-project

python -m inferences.qwen_sae_server \
  --base-model-path <path-to-qwen2.5-7b-instruct> \
  --sae-root <path-to-sae-checkpoints> \
  --layer 15 \
  --trainer 1 \
  --device cuda:0 \
  --topn 64 \
  --port 8000 \
  --log-path ~/anlp-fall2025-final-project/results/sae_server_logs_l15_k64.jsonl \
  --run-id sotopia_eval_l15_k64
```

## Run Sotopia eval locally

Terminal 1
```bash
redis-server
```

Terminal 2
```bash
export CUSTOM_API_KEY="your_litellm_api_key"
export OPENAI_API_KEY=""

cd ~/anlp-fall2025-final-project
./run_sotopia_eval.sh Qwen/Qwen2.5-7B-Instruct 8000
```
