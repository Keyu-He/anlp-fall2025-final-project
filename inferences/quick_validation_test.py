"""
Quick Validation Test for Feature Steering
快速验证 feature steering 是否有效的测试脚本

测试策略：
- 每个维度选择最强的 1-2 个 features
- 选择 5 个代表性场景
- 使用固定小强度 (1.0)
- 对比 baseline vs steered
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
        default="results/quick_validation",
    )
    return parser.parse_args()


# 测试配置：每个维度选最强的 features
TEST_CONFIGS = [
    # Relationship (最强相关性 0.5276)
    {"dimension": "relationship", "feature": 325, "strength": 1.0, "scenarios": [0, 2, 4]},
    {"dimension": "relationship", "feature": 543, "strength": 1.0, "scenarios": [0]},

    # Believability (最强负相关 -0.9182)
    {"dimension": "believability", "feature": 226, "strength": -1.0, "scenarios": [0, 1]},
    {"dimension": "believability", "feature": 226, "strength": 1.0, "scenarios": [1]},  # 测试反向

    # Knowledge (最强相关性 0.4882)
    {"dimension": "knowledge", "feature": 545, "strength": 1.0, "scenarios": [3]},
    {"dimension": "knowledge", "feature": 388, "strength": 1.0, "scenarios": [3]},

    # Financial (最强相关性 0.5490)
    {"dimension": "financial", "feature": 15, "strength": 1.0, "scenarios": [0, 3]},

    # Goal (最强负相关 -0.5070)
    {"dimension": "goal", "feature": 86, "strength": -1.0, "scenarios": [1]},
]


def load_sotopia_records(data_path: str):
    records = []
    with open(data_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_prompt_and_scenario(record, turn_idx=1):
    # Extract prompt
    if turn_idx >= len(record['meta']['messages']):
        turn_idx = len(record['meta']['messages']) - 1
    turn_messages = record['meta']['messages'][turn_idx]
    prompt = turn_messages[0][2]

    # Extract scenario
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


def check_repetition(text: str) -> dict:
    """检测输出重复情况"""
    words = text.split()
    if len(words) < 10:
        return {"has_repetition": False, "repetition_ratio": 0.0}

    # 检查连续重复词
    consecutive_repeats = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
    repeat_ratio = consecutive_repeats / len(words)

    # 检查短语重复
    phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    unique_phrases = len(set(phrases))
    phrase_repeat_ratio = 1 - (unique_phrases / len(phrases)) if phrases else 0

    has_repetition = repeat_ratio > 0.2 or phrase_repeat_ratio > 0.5

    return {
        "has_repetition": has_repetition,
        "consecutive_repeat_ratio": repeat_ratio,
        "phrase_repeat_ratio": phrase_repeat_ratio,
    }


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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"validation_results_{timestamp}.jsonl"
    summary_file = output_dir / f"summary_{timestamp}.md"

    print("=" * 80)
    print("快速验证测试：Feature Steering 是否有效？")
    print("=" * 80)
    print(f"\n测试策略：")
    print(f"  - 每个维度测试 1-2 个最强 features")
    print(f"  - 使用固定小强度 (1.0)")
    print(f"  - 对比 baseline vs steered 输出")
    print(f"  - 自动检测重复问题")
    print("=" * 80)

    # Load model
    print(f"\n加载模型: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        device_map=args.device,
    )
    model.eval()

    print(f"加载 SAE: {args.sae_dir}")
    ae, sae_cfg = dl_utils.load_dictionary(args.sae_dir, device=args.device)

    print(f"加载 Sotopia 数据: {args.sotopia_data}")
    records = load_sotopia_records(args.sotopia_data)

    # Expand test configs
    test_cases = []
    for config in TEST_CONFIGS:
        for scenario_idx in config["scenarios"]:
            test_cases.append({
                "dimension": config["dimension"],
                "feature": config["feature"],
                "strength": config["strength"],
                "scenario_idx": scenario_idx,
            })

    print(f"\n总测试数: {len(test_cases)}")
    print("=" * 80)

    results = []
    successful = 0
    has_repetition_count = 0

    with open(results_file, 'w', encoding='utf-8') as f_out:
        for i, test_case in enumerate(tqdm(test_cases, desc="测试进度"), 1):
            try:
                scenario_idx = test_case["scenario_idx"]
                feature_idx = test_case["feature"]
                strength = test_case["strength"]
                dimension = test_case["dimension"]

                record = records[scenario_idx]
                prompt, scenario = extract_prompt_and_scenario(record, turn_idx=1)

                baseline_text, steered_text, baseline_feat_val = generate_with_steering(
                    model, tokenizer, ae, prompt, args.layer, feature_idx,
                    strength, args.max_new_tokens, args.temperature, args.device,
                )

                # Check repetition
                baseline_rep = check_repetition(baseline_text)
                steered_rep = check_repetition(steered_text)

                if steered_rep["has_repetition"]:
                    has_repetition_count += 1

                # Check if outputs are different
                is_different = baseline_text.strip() != steered_text.strip()

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
                    "baseline_repetition": baseline_rep,
                    "steered_repetition": steered_rep,
                    "outputs_different": is_different,
                    "status": "success",
                }

                results.append(result)
                successful += 1

                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()

                # Print summary
                status_mark = "✓" if is_different else "="
                rep_mark = "⚠️" if steered_rep["has_repetition"] else ""
                print(f"\n[{i}/{len(test_cases)}] {status_mark} {dimension.upper()} (f{feature_idx}, s{strength}) {rep_mark}")
                print(f"  场景: {scenario[:60]}...")
                print(f"  Baseline: {baseline_text[:70]}...")
                print(f"  Steered:  {steered_text[:70]}...")
                if not is_different:
                    print(f"  ⚠️  输出相同，steering 可能无效")

            except Exception as e:
                print(f"\n[{i}/{len(test_cases)}] ✗ 失败: {str(e)}")
                result = {"test_idx": i, "status": "failed", "error": str(e), **test_case}
                results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()

    # Generate summary
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总测试数: {len(test_cases)}")
    print(f"成功: {successful}")
    print(f"失败: {len(test_cases) - successful}")
    print(f"有重复问题: {has_repetition_count}")

    # Count by dimension
    dimension_stats = {}
    for result in results:
        if result["status"] == "success":
            dim = result["dimension"]
            if dim not in dimension_stats:
                dimension_stats[dim] = {"total": 0, "different": 0, "repetition": 0}
            dimension_stats[dim]["total"] += 1
            if result["outputs_different"]:
                dimension_stats[dim]["different"] += 1
            if result["steered_repetition"]["has_repetition"]:
                dimension_stats[dim]["repetition"] += 1

    print("\n各维度统计:")
    for dim, stats in dimension_stats.items():
        print(f"  {dim:15s}: {stats['different']}/{stats['total']} 有效, "
              f"{stats['repetition']} 重复")

    # Write markdown summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Feature Steering 快速验证测试结果\n\n")
        f.write(f"**测试时间**: {timestamp}\n\n")
        f.write(f"## 总体统计\n\n")
        f.write(f"- 总测试数: {len(test_cases)}\n")
        f.write(f"- 成功: {successful}\n")
        f.write(f"- 失败: {len(test_cases) - successful}\n")
        f.write(f"- 有重复问题: {has_repetition_count}\n\n")

        f.write(f"## 各维度效果\n\n")
        f.write("| 维度 | 总数 | 有效数 | 有效率 | 重复数 |\n")
        f.write("|------|------|--------|--------|--------|\n")
        for dim, stats in sorted(dimension_stats.items()):
            eff_rate = stats['different'] / stats['total'] * 100 if stats['total'] > 0 else 0
            f.write(f"| {dim} | {stats['total']} | {stats['different']} | "
                   f"{eff_rate:.1f}% | {stats['repetition']} |\n")

        f.write(f"\n## 详细结果\n\n")
        for result in results:
            if result["status"] == "success":
                f.write(f"### Test {result['test_idx']}: {result['dimension']} "
                       f"(Feature {result['feature_idx']}, Strength {result['steering_strength']})\n\n")
                f.write(f"**场景**: {result['scenario']}\n\n")
                f.write(f"**Baseline Feature Value**: {result['baseline_feature_value']:.4f}\n\n")
                f.write(f"**输出是否不同**: {'✓ 是' if result['outputs_different'] else '✗ 否'}\n\n")
                if result['steered_repetition']['has_repetition']:
                    f.write(f"**⚠️ Steered 输出有重复问题**\n\n")
                f.write(f"**Baseline 输出**:\n```\n{result['baseline_output']}\n```\n\n")
                f.write(f"**Steered 输出**:\n```\n{result['steered_output']}\n```\n\n")
                f.write("---\n\n")

    print(f"\n详细结果已保存:")
    print(f"  - JSONL: {results_file}")
    print(f"  - Summary: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
