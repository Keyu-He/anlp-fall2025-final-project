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
        default="/data/user_data/keyuhe/qwen2.5-7b-sotopia/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1",
        help="Path to SAE directory. If path doesn't exist, downloads from HF.",
    )
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--feature-idx", type=int, default=None, help="Single feature index to interpret")
    parser.add_argument("--feature-indices-file", type=str, default=None, help="Path to text file with feature indices (one per line)")
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[-10.0, -5.0, 0.0, 5.0, 10.0],
        help="List of steering strengths to test (e.g. -10 -5 0 5 10)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["burning_trash", "borrow_money", "project_deadline"],
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Scenarios to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/interpretation",
        help="Directory to save results",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
    
    # Determine features to run
    feature_indices = []
    if args.feature_idx is not None:
        feature_indices.append(args.feature_idx)
    
    if args.feature_indices_file:
        if os.path.exists(args.feature_indices_file):
            with open(args.feature_indices_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line.isdigit():
                        feature_indices.append(int(line))
        else:
            print(f"Warning: Feature file {args.feature_indices_file} not found.")

    if not feature_indices:
        print("No features specified. Use --feature-idx or --feature-indices-file.")
        return

    print(f"Total features to process: {len(feature_indices)}")

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

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    from tqdm import tqdm
    for i, feature_idx in enumerate(tqdm(feature_indices, desc="Features")):
        # print(f"\n[{i+1}/{len(feature_indices)}] Processing Feature {feature_idx}...")
        
        # Check if already done
        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"feature_{feature_idx}_interpretation.json")
            if os.path.exists(output_path):
                print(f"Skipping Feature {feature_idx}, result already exists at {output_path}")
                continue

        all_results = {}

        for scenario_key in scenarios_to_run:
            prompt = SCENARIOS[scenario_key]
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            # print(f"  Scenario: {scenario_key}")
            
            scenario_results = {}
            
            for strength in args.strengths:
                # print(f"    Strength {strength}: Generating...", end="", flush=True)
                hook = None
                if abs(strength) > 1e-6:
                    hook = attach_steering_hook(model, args.layer, ae, feature_idx, strength)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                
                if hook:
                    hook.remove()
                    
                generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                print(f" Done ({len(generated)} chars)")
                # Print full output for validation
                print(f"      -> {generated}")
                scenario_results[strength] = generated
            
            all_results[scenario_key] = scenario_results
            
        # Automated Interpretation (this calls OpenAI API)
        print(f"Interpreting Feature {feature_idx}...")
        interpretation = interpret_results(feature_idx, all_results)
        
        # Save results immediately
        if args.output_dir:
            output_data = {
                "feature_idx": feature_idx,
                "scenarios": all_results,
                "interpretation": interpretation
            }
            
            import json
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Saved results to {output_path}")
        
        # Optional: Clear cache if running very large batches
        # torch.cuda.empty_cache()

def interpret_results(feature_idx, all_results):
    """
    Uses an LLM to interpret the feature based on generation differences across multiple scenarios.
    Returns a dictionary with 'title' and 'explanation'.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        print("OpenAI library not installed. Skipping automated interpretation.")
        return None
    except Exception as e:
        print(f"Could not initialize OpenAI client: {e}. Skipping automated interpretation.")
        return None

    prompt = f"""
You are an expert in interpreting Sparse Autoencoder (SAE) features in Language Models.
I will provide you with text generations from a model across multiple scenarios.
For each scenario, I will show:
1. Negative Steering: The feature was suppressed or steered negatively.
2. Baseline: The original model behavior.
3. Positive Steering: The feature was activated or steered positively.

Your task is to analyze the differences across ALL scenarios and find the COMMON underlying semantic meaning of SAE Feature {feature_idx}.
Do NOT focus on the specific topic of one scenario (e.g., if one scenario is about trash, don't just say "trash"). Look for the abstract behavior or concept that explains the changes in all scenarios.

"""

    for scenario, results in all_results.items():
        strengths = sorted(results.keys())
        if len(strengths) < 3:
            continue
            
        prompt += f"### Scenario: {scenario}\n"
        
        for strength in strengths:
            if strength == 0.0:
                label = "Baseline (Strength 0.0)"
            elif strength < 0:
                label = f"Negative Steering (Strength {strength})"
            else:
                label = f"Positive Steering (Strength {strength})"
            
            prompt += f"[{label}]\n{results[strength]}\n\n"

    prompt += f"""
Based on these outputs, what specific concept, behavior, or topic does Feature {feature_idx} represent?

You must provide the answer in the following exact format:
Title: [A concise title (3-5 words)]
Explanation: [A brief explanation that fits ALL scenarios]
"""
    
    print(f"\n{'='*40}")
    print("AUTOMATED INTERPRETATION (GPT-5)")
    print(f"{'='*40}")
    
    interpretation = None
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a helpful interpretability assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        content = response.choices[0].message.content
        print(content)
        print(f"{'='*80}\n")
        
        # Parse output
        title = ""
        explanation = ""
        
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.strip().startswith("Explanation:"):
                explanation = line.replace("Explanation:", "").strip()
        
        # Fallback if explanation spans multiple lines or parsing failed slightly
        if not explanation and "Explanation:" in content:
             explanation = content.split("Explanation:")[1].strip()
             
        interpretation = {
            "title": title,
            "explanation": explanation
        }
        
    except Exception as e:
        print(f"Error during API call: {e}")
        print("Prompt that would have been sent:")
        print(prompt)
        
    return interpretation

if __name__ == "__main__":
    main()
