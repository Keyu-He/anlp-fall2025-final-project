"""
Feature Steering with Real Sotopia Tasks
Load prompts from Sotopia dataset and test feature steering
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dictionary_learning import utils as dl_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature steering with Sotopia tasks"
    )
    parser.add_argument(
        "--sotopia-data",
        type=str,
        default="/home/demiw/anlp-fall2025-final-project/results/sotopia_all_gpt-4o_Qwen_Qwen2.5-7B-Instruct_20251201_merged.jsonl",
        help="Path to Sotopia jsonl data file",
    )
    parser.add_argument(
        "--record-idx",
        type=int,
        default=None,
        help="Specific record index to use (default: random)",
    )
    parser.add_argument(
        "--turn-idx",
        type=int,
        default=1,
        help="Which turn to use (default: 1)",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/data/user_data/demiw/qwen2.5-7b-sotopia/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--sae-dir",
        type=str,
        default="/data/user_data/demiw/qwen2.5-7b-sotopia/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1",
    )
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument(
        "--feature-idx",
        type=int,
        required=True,
        help="Feature index to steer",
    )
    parser.add_argument(
        "--steering-strength",
        type=float,
        default=5.0,
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-file", type=str, default=None)
    return parser.parse_args()


def extract_sotopia_prompt(
    data_path: str, record_idx: int = None, turn_idx: int = 1
) -> Tuple[str, Dict]:
    """Extract a prompt from Sotopia data."""
    records = []
    with open(data_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    if record_idx is None:
        record_idx = random.randint(0, len(records) - 1)

    if record_idx >= len(records):
        raise ValueError(f"Record index {record_idx} out of range (0-{len(records)-1})")

    record = records[record_idx]

    # Extract scenario info
    first_turn = record['meta']['messages'][0]
    context = first_turn[0][2]

    scenario_start = context.find("Scenario:") + len("Scenario:")
    scenario_end = context.find("Participants:")
    scenario = context[scenario_start:scenario_end].strip()

    # Extract the prompt for the specified turn
    if turn_idx >= len(record['meta']['messages']):
        turn_idx = len(record['meta']['messages']) - 1
        print(f"Warning: turn_idx adjusted to {turn_idx}")

    turn_messages = record['meta']['messages'][turn_idx]
    prompt = turn_messages[0][2]  # The environment message to the agent

    metadata = {
        'record_idx': record_idx,
        'turn_idx': turn_idx,
        'scenario': scenario,
        'environment_id': record['environment']['id'],
        'agent_names': record['environment']['agent_names'],
    }

    return prompt, metadata


def attach_resid_post_hook(model, layer_index: int, storage: Dict[str, torch.Tensor]):
    """Hook to capture resid_post at the specified layer."""
    layer_module = model.model.layers[layer_index]

    def hook(_module, inputs, output):
        pre_mlp = inputs[0]
        mlp_out = output
        resid_post = pre_mlp + mlp_out
        storage["resid_post"] = resid_post

    return layer_module.mlp.register_forward_hook(hook)


def attach_steering_hook(
    model, layer_index: int, ae, feature_idx: int, steering_strength: float, device: str
):
    """Hook that steers a specific SAE feature during generation."""
    layer_module = model.model.layers[layer_index]

    def intervention_hook(_module, inputs, output):
        pre_mlp = inputs[0]
        mlp_out = output
        resid_post = pre_mlp + mlp_out

        # Save original dtype
        original_dtype = resid_post.dtype
        batch_size, seq_len, d_model = resid_post.shape

        # Encode with SAE (need float32)
        resid_flat = resid_post.view(-1, d_model).to(dtype=torch.float32)
        reconstructed, features = ae(resid_flat, output_features=True)

        # Steer the target feature
        features[:, feature_idx] += steering_strength

        # Decode back
        steered_resid = ae.decode(features)
        steered_resid = steered_resid.view(batch_size, seq_len, d_model)

        # Convert back to original dtype (bfloat16)
        steered_resid = steered_resid.to(dtype=original_dtype)

        # Return modified MLP output
        return steered_resid - pre_mlp

    return layer_module.mlp.register_forward_hook(intervention_hook)


def main():
    args = parse_args()

    print("=" * 80)
    print("SAE Feature Steering with Sotopia Tasks")
    print("=" * 80)

    # Load Sotopia prompt
    print(f"\nLoading Sotopia task from: {args.sotopia_data}")
    prompt, metadata = extract_sotopia_prompt(
        args.sotopia_data, args.record_idx, args.turn_idx
    )

    print(f"\nTask metadata:")
    print(f"  Record: {metadata['record_idx']}")
    print(f"  Turn: {metadata['turn_idx']}")
    print(f"  Scenario: {metadata['scenario'][:100]}...")
    print(f"  Agents: {', '.join(metadata['agent_names'])}")

    print(f"\nSteering configuration:")
    print(f"  Feature index: {args.feature_idx}")
    print(f"  Steering strength: {args.steering_strength}")
    print(f"  Layer: {args.layer}")

    # Load model and tokenizer
    print(f"\nLoading model from: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        device_map=args.device,
    )
    model.eval()

    # Load SAE
    print(f"Loading SAE from: {args.sae_dir}")
    ae, sae_cfg = dl_utils.load_dictionary(args.sae_dir, device=args.device)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # ========== Baseline generation ==========
    print("\n" + "=" * 80)
    print("Baseline Generation (No Steering)")
    print("=" * 80)

    baseline_storage = {}
    baseline_hook = attach_resid_post_hook(model, args.layer, baseline_storage)

    with torch.no_grad():
        baseline_outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else 1.0,
        )

    baseline_hook.remove()

    baseline_text = tokenizer.decode(
        baseline_outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Analyze baseline features
    baseline_feature_val = None
    if "resid_post" in baseline_storage:
        resid = baseline_storage["resid_post"][0].to(args.device, dtype=torch.float32)
        _, features = ae(resid, output_features=True)
        feat_mean = features.mean(dim=0)
        baseline_feature_val = float(feat_mean[args.feature_idx].item())
        print(f"Baseline feature[{args.feature_idx}] value: {baseline_feature_val:.4f}")

    print(f"\nBaseline output:\n{baseline_text}")

    # ========== Steered generation ==========
    print("\n" + "=" * 80)
    print(f"Steered Generation (feature {args.feature_idx}, strength {args.steering_strength})")
    print("=" * 80)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    steered_hook = attach_steering_hook(
        model, args.layer, ae, args.feature_idx, args.steering_strength, args.device
    )

    with torch.no_grad():
        steered_outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else 1.0,
        )

    steered_hook.remove()

    steered_text = tokenizer.decode(
        steered_outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print(f"\nSteered output:\n{steered_text}")

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nPrompt (first 200 chars):\n{prompt[:200]}...")
    print(f"\n[Baseline] {baseline_text[:150]}...")
    print(f"\n[Steered]  {steered_text[:150]}...")

    # Save results
    if args.output_file:
        results = {
            'metadata': metadata,
            'steering': {
                'feature_idx': args.feature_idx,
                'steering_strength': args.steering_strength,
                'layer': args.layer,
            },
            'prompt': prompt,
            'baseline_output': baseline_text,
            'steered_output': steered_text,
            'baseline_feature_value': baseline_feature_val,
            'config': {
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature,
            },
        }

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {args.output_file}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
