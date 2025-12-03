"""
Feature Steering with SAE for Sotopia
Simple script to test single feature steering and compare outputs.
"""

import argparse
import json
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dictionary_learning import utils as dl_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature steering with SAE for Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/data/user_data/demiw/qwen2.5-7b-sotopia/Qwen2.5-7B-Instruct",
        help="Path to Qwen2.5-7B-Instruct weights",
    )
    parser.add_argument(
        "--sae-dir",
        type=str,
        default="/data/user_data/demiw/qwen2.5-7b-sotopia/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1",
        help="Path to SAE directory",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Layer index for steering",
    )
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
        help="Steering strength (positive to enhance, negative to suppress)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input prompt for inference",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file to save results (optional)",
    )
    return parser.parse_args()


def attach_resid_post_hook(model, layer_index: int, storage: Dict[str, torch.Tensor]):
    """Hook to capture resid_post at the specified layer."""
    layer_module = model.model.layers[layer_index]
    if not hasattr(layer_module, "mlp"):
        raise ValueError(f"Layer {layer_index} has no MLP, cannot hook resid_post")

    def hook(_module, inputs, output):
        pre_mlp = inputs[0]  # before MLP
        mlp_out = output  # MLP(pre_mlp)
        resid_post = pre_mlp + mlp_out
        storage["resid_post"] = resid_post

    return layer_module.mlp.register_forward_hook(hook)


def attach_steering_hook(model, layer_index: int, ae, feature_idx: int, steering_strength: float, device: str):
    """Hook that steers a specific SAE feature during generation."""
    layer_module = model.model.layers[layer_index]
    if not hasattr(layer_module, "mlp"):
        raise ValueError(f"Layer {layer_index} has no MLP, cannot hook resid_post")

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

    print("=" * 70)
    print("SAE Feature Steering for Qwen2.5-7B")
    print("=" * 70)
    print(f"Feature index: {args.feature_idx}")
    print(f"Steering strength: {args.steering_strength}")
    print(f"Layer: {args.layer}")
    print("=" * 70)

    # Load model and tokenizer
    print(f"\nLoading model from: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
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
    trainer_cfg = sae_cfg.get("trainer", {})
    print(f"SAE config: {trainer_cfg}")

    # Tokenize input
    inputs = tokenizer(args.text, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # ========== Baseline generation (no steering) ==========
    print("\n" + "=" * 70)
    print("Baseline Generation (No Steering)")
    print("=" * 70)

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

    # Analyze baseline SAE features
    baseline_feature_val = None
    if "resid_post" in baseline_storage:
        resid = baseline_storage["resid_post"][0].to(args.device, dtype=torch.float32)
        _, features = ae(resid, output_features=True)
        feat_mean = features.mean(dim=0)
        baseline_feature_val = float(feat_mean[args.feature_idx].item())
        print(f"Baseline feature[{args.feature_idx}] value: {baseline_feature_val:.4f}")

    print(f"\nBaseline output:\n{baseline_text}")

    # ========== Steered generation ==========
    print("\n" + "=" * 70)
    print(f"Steered Generation (feature {args.feature_idx}, strength {args.steering_strength})")
    print("=" * 70)

    # Re-tokenize for fresh generation
    inputs = tokenizer(args.text, return_tensors="pt")
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

    # Save results if output file specified
    if args.output_file:
        results = {
            "feature_idx": args.feature_idx,
            "steering_strength": args.steering_strength,
            "layer": args.layer,
            "prompt": args.text,
            "baseline_output": baseline_text,
            "steered_output": steered_text,
            "baseline_feature_value": baseline_feature_val,
            "config": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "sae_dir": args.sae_dir,
            },
        }

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {args.output_file}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()