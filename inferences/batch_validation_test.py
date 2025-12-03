"""
Batch Validation Test - 可配置的批量测试
支持自定义测试数量、强度范围等
"""

import argparse
import json
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
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/batch_validation",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=10,
        help="使用多少个场景（默认：10）",
    )
    parser.add_argument(
        "--features-per-dim",
        type=int,
        default=3,
        help="每个维度测试多少个 features（默认：3）",
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
        help="测试的强度列表（默认：0.5 1.0 1.5）",
    )
    return parser.parse_args()


# Feature 配置（按相关性排序）
FEATURES_CONFIG = {
    "relationship": [325, 543, 113, 116, 20],  # Top 5
    "believability": [226, 485, 93, 37, 379],
    "knowledge": [545, 388, 451, 78, 116],
    "financial": [15, 401, 209, 536, 219],
    "goal": [86, 531, 196, 379, 136],
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

    with torch.no_grad():
        baseline_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
        )

    baseline_text = tokenizer.decode(
        baseline_outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

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

    return baseline_text, steered_text


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"batch_results_{timestamp}.jsonl"

    print("=" * 80)
    print("批量验证测试")
    print("=" * 80)
    print(f"配置:")
    print(f"  场景数量: {args.num_scenarios}")
    print(f"  每维度 features: {args.features_per_dim}")
    print(f"  强度列表: {args.strengths}")
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
    scenario_indices = list(range(min(args.num_scenarios, len(records))))

    for dimension, features in FEATURES_CONFIG.items():
        # 选择 top N features
        selected_features = features[:args.features_per_dim]

        for feature in selected_features:
            for strength in args.strengths:
                for scenario_idx in scenario_indices:
                    test_cases.append({
                        "dimension": dimension,
                        "feature": feature,
                        "strength": strength,
                        "scenario_idx": scenario_idx,
                    })

    print(f"\n总测试数: {len(test_cases)}")
    print(f"预计时间: ~{len(test_cases) * 0.5:.0f} 分钟（每个测试约 30 秒）")
    print("=" * 80)

    results = []
    successful = 0

    with open(results_file, 'w', encoding='utf-8') as f_out:
        for i, test_case in enumerate(tqdm(test_cases, desc="测试进度"), 1):
            try:
                scenario_idx = test_case["scenario_idx"]
                feature_idx = test_case["feature"]
                strength = test_case["strength"]
                dimension = test_case["dimension"]

                record = records[scenario_idx]
                prompt, scenario = extract_prompt_and_scenario(record, turn_idx=1)

                baseline_text, steered_text = generate_with_steering(
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
                    "baseline_output": baseline_text,
                    "steered_output": steered_text,
                    "outputs_different": baseline_text.strip() != steered_text.strip(),
                    "status": "success",
                }

                results.append(result)
                successful += 1

                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()

            except Exception as e:
                print(f"\n✗ Test {i} 失败: {str(e)}")
                result = {"test_idx": i, "status": "failed", "error": str(e), **test_case}
                results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()

    # Summary
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    print(f"总测试数: {len(test_cases)}")
    print(f"成功: {successful}")
    print(f"失败: {len(test_cases) - successful}")

    different_count = sum(1 for r in results if r.get('outputs_different', False))
    print(f"输出有变化: {different_count} / {successful} = {different_count/successful*100 if successful > 0 else 0:.1f}%")

    print(f"\n结果已保存到: {results_file}")
    print(f"\n生成分析报告:")
    print(f"  python inferences/analyze_steering_results.py {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
