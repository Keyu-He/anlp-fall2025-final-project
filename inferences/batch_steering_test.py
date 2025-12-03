"""
Batch Feature Steering Test with Small Strengths
Test multiple scenarios and features with smaller steering strengths to avoid repetition
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from dictionary_learning import utils as dl_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sotopia-data",
        type=str,
        default="/home/demiw/anlp-fall2025-final-project/results/sotopia_all_gpt-4o_Qwen_Qwen2.5-7B-Instruct_20251201_merged.jsonl",
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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/batch_steering",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to test",
    )
    return parser.parse_args()


def load_sotopia_records(data_path: str):
    """Load all Sotopia records."""
    records = []
    with open(data_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_prompt(record, turn_idx=1):
    """Extract prompt from a record."""
    if turn_idx >= len(record['meta']['messages']):
        turn_idx = len(record['meta']['messages']) - 1
    turn_messages = record['meta']['messages'][turn_idx]
    return turn_messages[0][2]


def attach_resid_post_hook(model, layer_index: int, storage: Dict[str, torch.Tensor]):
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


def generate_with_steering(
    model,
    tokenizer,
    ae,
    prompt: str,
    layer: int,
    feature_idx: int,
    steering_strength: float,
    max_new_tokens: int,
    temperature: float,
    device: str,
):
    """Generate baseline and steered outputs."""
    # Baseline
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    baseline_storage = {}
    baseline_hook = attach_resid_post_hook(model, layer, baseline_storage)

    with torch.no_grad():
        baseline_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
        )

    baseline_hook.remove()

    baseline_text = tokenizer.decode(
        baseline_outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Get baseline feature value
    baseline_feature_val = None
    if "resid_post" in baseline_storage:
        resid = baseline_storage["resid_post"][0].to(device, dtype=torch.float32)
        _, features = ae(resid, output_features=True)
        feat_mean = features.mean(dim=0)
        baseline_feature_val = float(feat_mean[feature_idx].item())

    # Steered
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    steered_hook = attach_steering_hook(
        model, layer, ae, feature_idx, steering_strength, device
    )

    with torch.no_grad():
        steered_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
        )

    steered_hook.remove()

    steered_text = tokenizer.decode(
        steered_outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return baseline_text, steered_text, baseline_feature_val


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"batch_results_{timestamp}.jsonl"

    print("=" * 80)
    print("Batch Feature Steering Test (Small Strengths)")
    print("=" * 80)

    # Load model
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

    # Load Sotopia data
    print(f"Loading Sotopia data from: {args.sotopia_data}")
    records = load_sotopia_records(args.sotopia_data)
    print(f"Total records: {len(records)}")

    # Define test configurations
    # Using SMALL strengths to avoid repetition: 0.5, 1.0, 2.0
    test_configs = [
        # Relationship features with small positive strengths
        {"record_idx": 0, "feature": 325, "strength": 0.5, "dimension": "relationship"},
        {"record_idx": 0, "feature": 325, "strength": 1.0, "dimension": "relationship"},
        {"record_idx": 0, "feature": 325, "strength": 2.0, "dimension": "relationship"},
        {"record_idx": 0, "feature": 543, "strength": 1.0, "dimension": "relationship"},

        # Believability features with small negative strengths
        {"record_idx": 1, "feature": 226, "strength": -0.5, "dimension": "believability"},
        {"record_idx": 1, "feature": 226, "strength": -1.0, "dimension": "believability"},
        {"record_idx": 1, "feature": 226, "strength": -2.0, "dimension": "believability"},

        # Knowledge features with small positive strengths
        {"record_idx": 3, "feature": 545, "strength": 0.5, "dimension": "knowledge"},
        {"record_idx": 3, "feature": 545, "strength": 1.0, "dimension": "knowledge"},
        {"record_idx": 3, "feature": 388, "strength": 1.0, "dimension": "knowledge"},

        # Financial features
        {"record_idx": 0, "feature": 15, "strength": 0.5, "dimension": "financial"},
        {"record_idx": 0, "feature": 15, "strength": 1.0, "dimension": "financial"},

        # Goal features
        {"record_idx": 1, "feature": 86, "strength": -0.5, "dimension": "goal"},
        {"record_idx": 1, "feature": 86, "strength": -1.0, "dimension": "goal"},

        # Different scenarios with relationship
        {"record_idx": 2, "feature": 325, "strength": 1.0, "dimension": "relationship"},
        {"record_idx": 4, "feature": 325, "strength": 1.0, "dimension": "relationship"},
    ]

    # Limit to num_samples
    test_configs = test_configs[:args.num_samples]

    print(f"\nRunning {len(test_configs)} tests with small strengths (0.5-2.0)...")
    print("=" * 80)

    results = []
    successful = 0
    failed = 0

    with open(results_file, 'w', encoding='utf-8') as f_out:
        for i, config in enumerate(tqdm(test_configs, desc="Testing"), 1):
            try:
                record_idx = config["record_idx"]
                feature_idx = config["feature"]
                strength = config["strength"]
                dimension = config["dimension"]

                if record_idx >= len(records):
                    print(f"\nSkipping config {i}: record {record_idx} out of range")
                    continue

                record = records[record_idx]
                prompt = extract_prompt(record, turn_idx=1)

                # Get scenario info
                first_turn = record['meta']['messages'][0]
                context = first_turn[0][2]
                scenario_start = context.find("Scenario:") + len("Scenario:")
                scenario_end = context.find("Participants:")
                scenario = context[scenario_start:scenario_end].strip()

                baseline_text, steered_text, baseline_feat_val = generate_with_steering(
                    model=model,
                    tokenizer=tokenizer,
                    ae=ae,
                    prompt=prompt,
                    layer=args.layer,
                    feature_idx=feature_idx,
                    steering_strength=strength,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    device=args.device,
                )

                result = {
                    "test_idx": i,
                    "record_idx": record_idx,
                    "scenario": scenario[:150],
                    "dimension": dimension,
                    "feature_idx": feature_idx,
                    "steering_strength": strength,
                    "baseline_feature_value": baseline_feat_val,
                    "prompt": prompt[:200],
                    "baseline_output": baseline_text,
                    "steered_output": steered_text,
                    "status": "success",
                }

                # Check for repetition (simple heuristic)
                words = steered_text.split()
                if len(words) > 10:
                    # Check if many consecutive repeated words
                    repeated = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
                    if repeated > len(words) * 0.3:
                        result["warning"] = "high_repetition"

                results.append(result)
                successful += 1

                # Write to jsonl immediately
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()

                # Print progress
                print(f"\n[{i}/{len(test_configs)}] Record {record_idx}, Feature {feature_idx}, Strength {strength}")
                print(f"  Baseline: {baseline_text[:80]}...")
                print(f"  Steered:  {steered_text[:80]}...")

            except Exception as e:
                print(f"\n[{i}/{len(test_configs)}] FAILED: {str(e)}")
                failed += 1
                result = {
                    "test_idx": i,
                    "status": "failed",
                    "error": str(e),
                    **config,
                }
                results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(test_configs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {results_file}")

    # Create summary file
    summary_file = output_dir / f"summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Batch Steering Test Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total tests: {len(test_configs)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")

        f.write("Test Configurations:\n")
        f.write("-" * 80 + "\n")
        for i, result in enumerate(results, 1):
            if result["status"] == "success":
                f.write(f"\n{i}. Record {result['record_idx']}, "
                       f"Feature {result['feature_idx']}, "
                       f"Strength {result['steering_strength']}, "
                       f"Dimension: {result['dimension']}\n")
                f.write(f"   Scenario: {result['scenario']}\n")
                f.write(f"   Baseline feature value: {result.get('baseline_feature_value', 'N/A')}\n")
                if 'warning' in result:
                    f.write(f"   WARNING: {result['warning']}\n")
                f.write(f"   Baseline: {result['baseline_output'][:100]}...\n")
                f.write(f"   Steered:  {result['steered_output'][:100]}...\n")

    print(f"Summary saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
