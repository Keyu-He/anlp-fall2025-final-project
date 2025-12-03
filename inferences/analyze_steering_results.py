"""
Analyze Steering Results
分析 steering 测试结果，生成可视化报告
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_file",
        type=str,
        help="JSONL 结果文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出报告文件路径（默认：自动生成）",
    )
    return parser.parse_args()


def load_results(file_path: str) -> List[Dict]:
    """加载 JSONL 结果文件"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_by_dimension(results: List[Dict]) -> Dict:
    """按维度分析结果"""
    stats = defaultdict(lambda: {
        "total": 0,
        "successful": 0,
        "different": 0,
        "baseline_features": [],
    })

    for r in results:
        if r.get("status") != "success":
            continue

        dim = r["dimension"]
        stats[dim]["total"] += 1
        stats[dim]["successful"] += 1

        if r.get("outputs_different", False):
            stats[dim]["different"] += 1

        if r.get("baseline_feature_value") is not None:
            stats[dim]["baseline_features"].append(r["baseline_feature_value"])

    return dict(stats)


def analyze_by_feature(results: List[Dict]) -> Dict:
    """按 feature 分析结果"""
    stats = defaultdict(lambda: {
        "dimension": "",
        "total": 0,
        "different": 0,
        "strengths": [],
        "baseline_values": [],
    })

    for r in results:
        if r.get("status") != "success":
            continue

        feature = r["feature_idx"]
        stats[feature]["dimension"] = r["dimension"]
        stats[feature]["total"] += 1

        if r.get("outputs_different", False):
            stats[feature]["different"] += 1

        stats[feature]["strengths"].append(r["steering_strength"])

        if r.get("baseline_feature_value") is not None:
            stats[feature]["baseline_values"].append(r["baseline_feature_value"])

    return dict(stats)


def analyze_by_strength(results: List[Dict]) -> Dict:
    """按强度分析结果"""
    stats = defaultdict(lambda: {
        "total": 0,
        "different": 0,
    })

    for r in results:
        if r.get("status") != "success":
            continue

        strength = r["steering_strength"]
        stats[strength]["total"] += 1

        if r.get("outputs_different", False):
            stats[strength]["different"] += 1

    return dict(stats)


def find_best_cases(results: List[Dict], top_n: int = 5) -> List[Dict]:
    """找出最有效的测试案例"""
    successful = [r for r in results if r.get("status") == "success" and r.get("outputs_different", False)]

    # 按照 baseline feature value 排序（激活值高的可能更容易看出效果）
    successful_sorted = sorted(
        successful,
        key=lambda x: abs(x.get("baseline_feature_value", 0)),
        reverse=True
    )

    return successful_sorted[:top_n]


def generate_report(results: List[Dict], output_file: str):
    """生成分析报告"""
    dim_stats = analyze_by_dimension(results)
    feature_stats = analyze_by_feature(results)
    strength_stats = analyze_by_strength(results)
    best_cases = find_best_cases(results, top_n=10)

    total = len(results)
    successful = sum(1 for r in results if r.get("status") == "success")
    different = sum(1 for r in results if r.get("outputs_different", False))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Feature Steering 测试结果分析\n\n")

        # 总体统计
        f.write("## 总体统计\n\n")
        f.write(f"- **总测试数**: {total}\n")
        f.write(f"- **成功测试**: {successful} ({successful/total*100:.1f}%)\n")
        f.write(f"- **输出有变化**: {different} ({different/successful*100 if successful > 0 else 0:.1f}% of successful)\n")
        f.write(f"- **有效率**: {different/total*100:.1f}%\n\n")

        # 按维度统计
        f.write("## 各维度效果分析\n\n")
        f.write("| 维度 | 测试数 | 成功数 | 有变化 | 有效率 | 平均 Baseline 激活 |\n")
        f.write("|------|--------|--------|--------|--------|--------------------|\n")

        for dim in sorted(dim_stats.keys()):
            stats = dim_stats[dim]
            eff_rate = stats["different"] / stats["total"] * 100 if stats["total"] > 0 else 0
            avg_baseline = sum(stats["baseline_features"]) / len(stats["baseline_features"]) if stats["baseline_features"] else 0
            f.write(f"| {dim} | {stats['total']} | {stats['successful']} | "
                   f"{stats['different']} | {eff_rate:.1f}% | {avg_baseline:.4f} |\n")

        f.write("\n")

        # 按 Feature 统计
        f.write("## Top Features 效果排名\n\n")
        f.write("| Feature | 维度 | 测试数 | 有变化 | 有效率 | 平均 Baseline 激活 |\n")
        f.write("|---------|------|--------|--------|--------|--------------------|\ n")

        feature_sorted = sorted(
            feature_stats.items(),
            key=lambda x: x[1]["different"] / x[1]["total"] if x[1]["total"] > 0 else 0,
            reverse=True
        )

        for feature, stats in feature_sorted[:15]:  # Top 15
            eff_rate = stats["different"] / stats["total"] * 100 if stats["total"] > 0 else 0
            avg_baseline = sum(stats["baseline_values"]) / len(stats["baseline_values"]) if stats["baseline_values"] else 0
            f.write(f"| {feature} | {stats['dimension']} | {stats['total']} | "
                   f"{stats['different']} | {eff_rate:.1f}% | {avg_baseline:.4f} |\n")

        f.write("\n")

        # 按强度统计
        f.write("## Steering 强度效果分析\n\n")
        f.write("| 强度 | 测试数 | 有变化 | 有效率 |\n")
        f.write("|------|--------|--------|--------|\n")

        strength_sorted = sorted(strength_stats.items(), key=lambda x: x[0])
        for strength, stats in strength_sorted:
            eff_rate = stats["different"] / stats["total"] * 100 if stats["total"] > 0 else 0
            f.write(f"| {strength:+.1f} | {stats['total']} | {stats['different']} | {eff_rate:.1f}% |\n")

        f.write("\n")

        # 最佳案例
        f.write("## Top 10 最有效的测试案例\n\n")
        f.write("（按 baseline feature 激活值排序）\n\n")

        for i, case in enumerate(best_cases, 1):
            f.write(f"### {i}. {case['dimension'].upper()} - Feature {case['feature_idx']} "
                   f"(Strength {case['steering_strength']:+.1f})\n\n")
            f.write(f"**场景**: {case['scenario']}\n\n")
            f.write(f"**Baseline Feature 激活值**: {case.get('baseline_feature_value', 0):.4f}\n\n")
            f.write(f"**Baseline 输出**:\n```\n{case['baseline_output'][:200]}...\n```\n\n")
            f.write(f"**Steered 输出**:\n```\n{case['steered_output'][:200]}...\n```\n\n")
            f.write("---\n\n")

        # 建议
        f.write("## 分析建议\n\n")

        # 找出最有效的维度
        best_dim = max(dim_stats.items(), key=lambda x: x[1]["different"] / x[1]["total"] if x[1]["total"] > 0 else 0)
        f.write(f"1. **最有效的维度**: {best_dim[0]} "
               f"({best_dim[1]['different']}/{best_dim[1]['total']} = "
               f"{best_dim[1]['different']/best_dim[1]['total']*100:.1f}% 有效率)\n\n")

        # 找出最有效的强度范围
        if strength_sorted:
            best_strengths = [s for s, stats in strength_sorted if stats["different"] / stats["total"] > 0.5]
            if best_strengths:
                f.write(f"2. **推荐的强度范围**: {min(best_strengths):.1f} 到 {max(best_strengths):.1f}\n\n")

        # 找出最有效的 features
        if feature_sorted:
            top3_features = [f for f, stats in feature_sorted[:3]]
            f.write(f"3. **最有效的 Features**: {', '.join(map(str, top3_features))}\n\n")

        f.write(f"4. **建议**: \n")
        if different / successful < 0.3:
            f.write(f"   - 有效率较低（{different/successful*100:.1f}%），可能需要：\n")
            f.write(f"     - 尝试更大的 steering strengths\n")
            f.write(f"     - 选择 baseline 激活值更高的 features\n")
            f.write(f"     - 选择更合适的场景\n")
        elif different / successful > 0.7:
            f.write(f"   - 有效率很高（{different/successful*100:.1f}%），steering 效果明显！\n")
            f.write(f"   - 可以进一步分析输出质量和重复问题\n")
        else:
            f.write(f"   - 有效率中等（{different/successful*100:.1f}%），部分 features 有效\n")
            f.write(f"   - 建议重点关注有效率高的 features 和维度\n")

    print(f"分析报告已生成: {output_file}")


def main():
    args = parse_args()

    print(f"加载结果文件: {args.results_file}")
    results = load_results(args.results_file)
    print(f"总共 {len(results)} 条记录")

    if args.output:
        output_file = args.output
    else:
        # 自动生成输出文件名
        input_path = Path(args.results_file)
        output_file = input_path.parent / f"{input_path.stem}_analysis.md"

    print(f"生成分析报告...")
    generate_report(results, str(output_file))

    print("\n分析完成！")
    print(f"报告文件: {output_file}")


if __name__ == "__main__":
    main()
