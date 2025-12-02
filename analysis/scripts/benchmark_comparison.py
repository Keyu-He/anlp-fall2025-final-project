"""
Benchmark Comparison Analysis
Compares model performance across different Sotopia benchmarks
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Score ranges for normalization
SCORE_RANGES = {
    'believability': (0, 10),
    'knowledge': (0, 10),
    'goal': (0, 10),
    'relationship': (-5, 5),
    'financial_and_material_benefits': (-5, 5),
    'secret': (-10, 0),
    'social_rules': (-10, 0)
}

def normalize_score(score, metric_name):
    """Normalize score to 0-1 scale"""
    min_val, max_val = SCORE_RANGES.get(metric_name, (0, 10))
    return (score - min_val) / (max_val - min_val) if max_val > min_val else 0

def load_benchmark_data(file_path, benchmark_name, model_name):
    """Load and process benchmark data"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    rows = []
    for item in data:
        rewards = item['rewards']

        # Skip failed scenarios (rewards are simple floats)
        if not isinstance(rewards[0], (list, tuple)):
            continue

        # Extract metrics for agent2 only (the model being evaluated)
        # agent_idx=0 is gpt-4o (accompanying model)
        # agent_idx=1 is qwen-2.5-7b (model being evaluated)
        for agent_idx, reward_data in enumerate(rewards):
            # Only analyze agent2 (the model being evaluated)
            if agent_idx != 1:
                continue

            overall_score, metrics = reward_data

            row = {
                'benchmark': benchmark_name,
                'model': model_name,
                'scenario_id': item['environment']['id'],
                'agent_idx': agent_idx,
                'overall_score': overall_score,
            }

            # Add raw scores
            for metric_name in SCORE_RANGES.keys():
                row[metric_name] = metrics[metric_name]
                # Add normalized scores
                row[f'{metric_name}_norm'] = normalize_score(metrics[metric_name], metric_name)

            rows.append(row)

    return pd.DataFrame(rows)

# ============================================================================
# Load Data
# ============================================================================
print("="*80)
print("BENCHMARK COMPARISON ANALYSIS")
print("="*80)

# Define benchmarks to analyze
# Format: (file_path, benchmark_name, model_name)
benchmarks = [
    ('../results/sotopia_all_gpt-4o_qwen-2.5-7b-instruct.jsonl',
     'Sotopia All',
     'gpt-4o + qwen-2.5-7b'),
    ('../results/sotopia_hard_gpt-4o_qwen-2.5-7b-instruct_1.jsonl',
     'Sotopia Hard',
     'gpt-4o + qwen-2.5-7b'),
]

# Load all benchmarks
dfs = []
for file_path, benchmark_name, model_name in benchmarks:
    df = load_benchmark_data(file_path, benchmark_name, model_name)
    dfs.append(df)
    print(f"\n✓ Loaded {benchmark_name}: {len(df)} agent evaluations")

df_all = pd.concat(dfs, ignore_index=True)

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("AVERAGE SCORES BY BENCHMARK")
print("="*80)

metrics = list(SCORE_RANGES.keys())

# Calculate raw score averages
raw_summary = df_all.groupby('benchmark')[metrics + ['overall_score']].mean()
print("\nRaw Scores:")
print(raw_summary.round(3))

# Calculate normalized score averages
norm_metrics = [f'{m}_norm' for m in metrics]
norm_summary = df_all.groupby('benchmark')[norm_metrics].mean()
norm_summary.columns = metrics  # Rename for display
print("\nNormalized Scores (0-1 scale):")
print(norm_summary.round(3))

# Save to CSV
raw_summary.to_csv('benchmark_raw_scores.csv')
norm_summary.to_csv('benchmark_normalized_scores.csv')
print("\n✓ Saved: benchmark_raw_scores.csv")
print("✓ Saved: benchmark_normalized_scores.csv")

# ============================================================================
# Visualization 1: Normalized Scores Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Benchmark Comparison: gpt-4o + qwen-2.5-7b',
             fontsize=16, fontweight='bold', y=1.02)

# Plot 1: Bar chart of normalized scores
norm_summary_plot = norm_summary.T
norm_summary_plot.plot(kind='bar', ax=axes[0], width=0.7, alpha=0.8)
axes[0].set_xlabel('Metric', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
axes[0].set_title('Average Normalized Scores by Metric', fontsize=13, fontweight='bold')
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
axes[0].legend(title='Benchmark', fontsize=10, title_fontsize=11)
axes[0].tick_params(axis='x', rotation=45, labelsize=10)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim(0, 1.05)

# Add value labels on bars
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.2f', fontsize=8, padding=3)

# Plot 2: Overall score comparison
overall_scores = df_all.groupby('benchmark')['overall_score'].agg(['mean', 'std', 'count'])
x_pos = np.arange(len(overall_scores))
colors = ['#FF6B6B', '#4ECDC4']

bars = axes[1].bar(x_pos, overall_scores['mean'],
                   yerr=overall_scores['std'],
                   capsize=8, alpha=0.8, color=colors,
                   edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Benchmark', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Overall Score', fontsize=12, fontweight='bold')
axes[1].set_title('Overall Score Comparison', fontsize=13, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(overall_scores.index, fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val, std) in enumerate(zip(bars, overall_scores['mean'], overall_scores['std'])):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                f'{val:.3f}\n(±{std:.3f})', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

# Add sample size annotations
for i, (bar, count) in enumerate(zip(bars, overall_scores['count'])):
    axes[1].text(bar.get_x() + bar.get_width()/2, 0.1,
                f'n={count}', ha='center', va='bottom',
                fontsize=9, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: benchmark_comparison.png")

# ============================================================================
# Visualization 2: Detailed Metric Comparison
# ============================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('Detailed Metric Comparison: Sotopia All vs Sotopia Hard',
             fontsize=16, fontweight='bold')

axes = axes.flatten()

for idx, metric in enumerate(metrics):
    metric_norm = f'{metric}_norm'

    # Group by benchmark
    benchmark_data = []
    labels = []
    for benchmark in df_all['benchmark'].unique():
        data = df_all[df_all['benchmark'] == benchmark][metric_norm].values
        benchmark_data.append(data)
        labels.append(benchmark)

    # Create box plot
    bp = axes[idx].boxplot(benchmark_data, labels=labels, patch_artist=True,
                           widths=0.6, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    # Color the boxes
    colors_box = ['#FF6B6B', '#4ECDC4']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[idx].set_ylabel('Normalized Score (0-1)', fontsize=10)
    axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)
    axes[idx].set_ylim(-0.05, 1.05)
    axes[idx].tick_params(axis='x', labelsize=9)

    # Add mean values as text
    for i, (label, data) in enumerate(zip(labels, benchmark_data)):
        mean_val = np.mean(data)
        axes[idx].text(i+1, -0.02, f'{mean_val:.2f}',
                      ha='center', va='top', fontsize=9, fontweight='bold')

# Hide the last subplot (we have 7 metrics, 8 subplots)
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('detailed_metric_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: detailed_metric_comparison.png")

# ============================================================================
# Performance Difference Analysis
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE DIFFERENCES (Sotopia All - Sotopia Hard)")
print("="*80)

all_scores = df_all[df_all['benchmark'] == 'Sotopia All'][norm_metrics].mean()
hard_scores = df_all[df_all['benchmark'] == 'Sotopia Hard'][norm_metrics].mean()
diff = all_scores - hard_scores
diff.index = metrics

print("\nNormalized Score Differences:")
print(diff.sort_values(ascending=False).round(3))

# Visualization 3: Difference plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in diff.sort_values()]
diff.sort_values().plot(kind='barh', ax=ax, color=colors, width=0.7, edgecolor='black', linewidth=1.5)
ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.set_xlabel('Score Difference (Sotopia All - Sotopia Hard)', fontsize=12, fontweight='bold')
ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
ax.set_title('Performance Gap: Where Does Sotopia All Outperform/Underperform Hard?',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (metric, value) in enumerate(diff.sort_values().items()):
    ax.text(value + 0.01 if value > 0 else value - 0.01, i,
           f'{value:.3f}', va='center',
           ha='left' if value > 0 else 'right',
           fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_difference.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: benchmark_difference.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. benchmark_comparison.png - Main comparison chart")
print("  2. detailed_metric_comparison.png - Box plots for each metric")
print("  3. benchmark_difference.png - Performance gap analysis")
print("  4. benchmark_raw_scores.csv - Raw score summary")
print("  5. benchmark_normalized_scores.csv - Normalized score summary")
print("\n📊 Score Scales:")
for metric, (min_val, max_val) in SCORE_RANGES.items():
    print(f"  {metric}: {min_val} to {max_val}")
print("="*80)
