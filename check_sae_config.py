from huggingface_hub import hf_hub_download
import json

repo_id = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
filename = "layers.10.mlp/cfg.json"

try:
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(path, 'r') as f:
        cfg = json.load(f)
    print(json.dumps(cfg, indent=2))
except Exception as e:
    print(f"Error: {e}")
