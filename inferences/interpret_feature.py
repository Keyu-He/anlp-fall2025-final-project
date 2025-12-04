"""
Interpret SAE features by comparing generation with different steering strengths.
Usage: python inferences/interpret_feature.py --feature-idx 92506 --strengths -10 0 10
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary_learning import utils as dl_utils
from huggingface_hub import hf_hub_download

SCENARIOS = {
    "burning_trash": """Scenario: Mia Davis noticed William Brown burning trash, which violates community rules and is dangerous. Mia confronts William about it.
Participants: Mia Davis and William Brown
Mia Davis: "William, I noticed you were burning trash earlier. I understand it might seem like a quick way to get rid of it, but it's actually quite dangerous and could violate community rules. Burning trash can release harmful chemicals into the air, affecting our health and the environment. Plus, it could pose a fire hazard. I’m sure there are safer ways to handle your trash, and I'd be happy to help you find solutions."
William Brown:""",
    "borrow_money": """Scenario: Alex needs to borrow money from their friend Jordan to pay rent, but Jordan has been hesitant to lend money in the past.
Participants: Alex and Jordan
Alex: "Hey Jordan, I know things have been tight lately, but I'm in a really tough spot with rent this month. I was wondering if you could possibly lend me $200? I promise to pay you back as soon as I get my paycheck next week."
Jordan:""",
    "project_deadline": """Scenario: Sarah and Mike are working on a group project. Mike hasn't done his part, and the deadline is tomorrow. Sarah is frustrated.
Participants: Sarah and Mike
Sarah: "Mike, we need to talk. The project is due tomorrow, and I haven't seen any of your contributions yet. I've done my part, but we can't submit it like this. What's going on?"
Mike:"""
}

def parse_args():
    parser = argparse.ArgumentParser(description="Interpret SAE Feature")
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Path to Qwen2.5-7B-Instruct weights",
    )
    parser.add_argument(
        "--sae-dir",
        type=str,
        default=None,
        help="Path to SAE directory. If None, downloads from HF.",
    )
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--feature-idx", type=int, required=True)
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[-10.0, -5.0, 0.0, 5.0, 10.0],
        help="List of steering strengths to test (e.g. -10 0 10)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["burning_trash"],
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Scenarios to run",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps"if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()

def get_sae_dir(sae_dir_arg):
    if sae_dir_arg is not None and os.path.exists(sae_dir_arg):
        return sae_dir_arg
    
    print("Downloading SAE from Hugging Face (andyrdt/saes-qwen2.5-7b-instruct)...")
    repo_id = "andyrdt/saes-qwen2.5-7b-instruct"
    subfolder = "resid_post_layer_15/trainer_1"
    
    try:
        ae_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/ae.pt")
        hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/config.json")
        return os.path.dirname(ae_path)
    except Exception as e:
        print(f"Error downloading SAE: {e}")
        raise e

def attach_steering_hook(model, layer_index, ae, feature_idx, steering_strength):
    layer_module = model.model.layers[layer_index]
    
    def intervention_hook(_module, inputs, output):
        pre_mlp = inputs[0]
        mlp_out = output
        resid_post = pre_mlp + mlp_out
        
        original_dtype = resid_post.dtype
        batch_size, seq_len, d_model = resid_post.shape
        
        resid_flat = resid_post.view(-1, d_model).to(dtype=torch.float32)
        reconstructed, features = ae(resid_flat, output_features=True)
        
        features[:, feature_idx] += steering_strength
        
        steered_resid = ae.decode(features)
        steered_resid = steered_resid.view(batch_size, seq_len, d_model)
        steered_resid = steered_resid.to(dtype=original_dtype)
        
        return steered_resid - pre_mlp

    return layer_module.mlp.register_forward_hook(intervention_hook)

def main():
    args = parse_args()
    
    sae_dir = get_sae_dir(args.sae_dir)
    print(f"Using SAE directory: {sae_dir}")
    
    print(f"Loading model from {args.base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        device_map=args.device,
    )
    model.eval()

    print(f"Loading SAE from {sae_dir}...")
    ae, _ = dl_utils.load_dictionary(sae_dir, device=args.device)

    scenarios_to_run = args.scenarios
    if "all" in scenarios_to_run:
        scenarios_to_run = list(SCENARIOS.keys())

    print(f"\n{'='*80}")
    print(f"INTERPRETING FEATURE {args.feature_idx}")
    print(f"{'='*80}\n")

    for scenario_key in scenarios_to_run:
        prompt = SCENARIOS[scenario_key]
        print(f"Scenario: {scenario_key}")
        print(f"Prompt: {prompt[:100]}...\n")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        for strength in args.strengths:
            label = f"Strength {strength}"
            if strength == 0:
                label += " (Baseline)"
            
            print(f"--- {label} ---")
            
            hook = None
            if abs(strength) > 1e-6:
                hook = attach_steering_hook(model, args.layer, ae, args.feature_idx, strength)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            
            if hook:
                hook.remove()
                
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"{generated}\n")
        
        print(f"{'-'*80}\n")

if __name__ == "__main__":
    main()
