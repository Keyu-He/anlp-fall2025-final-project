"""
Demo script to test SAE feature steering for specific features.
Usage: python inferences/demo_steering.py --feature-idx 92506 --text "Your prompt here"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary_learning import utils as dl_utils

import os
from huggingface_hub import hf_hub_download

def parse_args():
    parser = argparse.ArgumentParser(description="Demo SAE Feature Steering")
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
        help="Path to SAE directory. If None, downloads from andyrdt/saes-qwen2.5-7b-instruct",
    )
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--feature-idx", type=int, default=92506)
    parser.add_argument("--steering-strength", type=float, default=10.0)
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input prompt (overrides scenario)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="burning_trash",
        choices=["burning_trash", "borrow_money", "project_deadline", "custom"],
        help="Preset scenario to use",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    return parser.parse_args()

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

def get_sae_dir(sae_dir_arg):
    if sae_dir_arg is not None and os.path.exists(sae_dir_arg):
        return sae_dir_arg
    
    print("Downloading SAE from Hugging Face (andyrdt/saes-qwen2.5-7b-instruct)...")
    repo_id = "andyrdt/saes-qwen2.5-7b-instruct"
    subfolder = "resid_post_layer_15/trainer_1"
    
    # Download ae.pt and config.json
    try:
        ae_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/ae.pt")
        config_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/config.json")
        
        # Return the directory containing the downloaded files
        return os.path.dirname(ae_path)
    except Exception as e:
        print(f"Error downloading SAE: {e}")
        print("Trying 'trainer_1' subfolder structure if applicable...")
        # Fallback or re-raise
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
    
    print(f"Steering Feature {args.feature_idx} with strength {args.steering_strength}")

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

    if args.text:
        prompt = args.text
    elif args.scenario in SCENARIOS:
        prompt = SCENARIOS[args.scenario]
    else:
        prompt = "The goal of the game is to"

    print(f"\nPrompt:\n{prompt}\n")
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    # Baseline
    print("\n--- Baseline Generation ---")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

    # Steered
    print(f"\n--- Steered Generation (Feature {args.feature_idx}) ---")
    hook = attach_steering_hook(model, args.layer, ae, args.feature_idx, args.steering_strength)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    hook.remove()
    print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

if __name__ == "__main__":
    main()
