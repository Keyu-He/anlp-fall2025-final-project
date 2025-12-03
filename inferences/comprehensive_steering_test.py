"""
Comprehensive Feature Steering Test
全面的 feature steering 测试：多强度、多场景、多 features

用于深入分析 steering 的效果和影响
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
        default="results/comprehensive_steering",
    )
    parser.add_argument(
        "--dimension",
        type=str,
        choices=["relationship", "believability", "knowledge", "financial", "goal", "all"],
        default="all",
        help="测试哪个维度（all=所有维度）",
    )
    return parser.parse_args()


# 全面测试配置
COMPREHENSIVE_CONFIGS = {
    "relationship": {
        "features": [325, 543, 113],  # Top 3
        "strengths": [0.5, 1.0, 1.5, 2.0],
        "scenarios": [0, 2, 4, 5],  # 议价、社交、浪漫等
    },
    "believability": {
        "features": [226, 485, 93],
        "strengths": [-2.0, -1.0, -0.5, 0.5, 1.0],  # 负相关，测试正负两个方向
        "scenarios": [0, 1, 2],
    },
    "knowledge": {
        "features": [545, 388, 451],
        "strengths": [0.5, 1.0, 1.5],
        "scenarios": [3, 5],  # 商业讨论等
    },
    "financial": {
        "features": [15, 401],  # 一个正相关一个负相关
        "strengths": [0.5, 1.0, 1.5],
        "scenarios": [0, 3],  # 议价、商业
    },
    "goal": {
        "features": [86, 531],
        "strengths": [-1.5, -1.0, -0.5],  # 负相关
        "scenarios": [1, 3],  # 囚徒困境、商业
    },
}


def load_sotopia_records(data_path: str):
    records = []
    with open(data_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_prompt_and_scenario(record, turn_idx=1):
    if turn_idx >= len(record['meta']['messages']):
        turn_idx = len(record['meta']['messages']) - 1
    turn_messages = record['meta']['messages'][turn_idx]
    prompt = turn_messages[0][2]

    first_turn = record['meta']['messages'][0]
    context = first_turn[0][2]
    scenario_start = context.find("Scenario:") + len("Scenario:")
    scenario_end = context.find("Participants:")
    scenario = context[scenario_start:scenario_end].strip()

    return prompt, scenario


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
    model, tokenizer, ae, prompt: str, layer: int, feature_idx: int,
    steering_strength: float, max_new_tokens: int, temperature: float, device: str,
):
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"comprehensive_{args.dimension}_{timestamp}.jsonl"

    print("=" * 80)
    print("全面 Feature Steering 测试")
    print("=" * 80)
    print(f"测试维度: {args.dimension}")
    print("=" * 80)

    # Load model
    print(f"\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        device_map=args.device,
    )
    model.eval()

    ae, sae_cfg = dl_utils.load_dictionary(args.sae_dir, device=args.device)
    records = load_sotopia_records(args.sotopia_data)

    # Generate test cases
    test_cases = []
    dimensions_to_test = list(COMPREHENSIVE_CONFIGS.keys()) if args.dimension == "all" else [args.dimension]

    for dimension in dimensions_to_test:
        config = COMPREHENSIVE_CONFIGS[dimension]
        for feature in config["features"]:
            for strength in config["strengths"]:
                for scenario_idx in config["scenarios"]:
                    test_cases.append({
                        "dimension": dimension,
                        "feature": feature,
                        "strength": strength,
                        "scenario_idx": scenario_idx,
                    })

    print(f"\n总测试数: {len(test_cases)}")
    print("=" * 80)

    results = []
    with open(results_file, 'w', encoding='utf-8') as f_out:
        for i, test_case in enumerate(tqdm(test_cases, desc="测试进度"), 1):
            try:
                scenario_idx = test_case["scenario_idx"]
                feature_idx = test_case["feature"]
                strength = test_case["strength"]
                dimension = test_case["dimension"]

                if scenario_idx >= len(records):
                    continue

                record = records[scenario_idx]
                prompt, scenario = extract_prompt_and_scenario(record, turn_idx=1)

                baseline_text, steered_text, baseline_feat_val = generate_with_steering(
                    model, tokenizer, ae, prompt, args.layer, feature_idx,
                    strength, args.max_new_tokens, args.temperature, args.device,
                )

                result = {
                    "test_idx": i,
                    "dimension": dimension,
                    "feature_idx": feature_idx,
                    "steering_strength": strength,
                    "scenario_idx": scenario_idx,
                    "scenario": scenario[:100],
                    "baseline_feature_value": baseline_feat_val,
                    "baseline_output": baseline_text,
                    "steered_output": steered_text,
                    "outputs_different": baseline_text.strip() != steered_text.strip(),
                    "status": "success",
                }

                results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()

            except Exception as e:
                print(f"\n✗ Test {i} 失败: {str(e)}")
                result = {"test_idx": i, "status": "failed", "error": str(e), **test_case}
                results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()

    print(f"\n测试完成！结果已保存到: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
