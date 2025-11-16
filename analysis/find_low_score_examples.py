"""
Find examples with very low scores on specific metrics
"""

import json
import pandas as pd

def load_and_analyze(file_path, benchmark_name):
    """Load data and find low score examples"""
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    examples = []

    for item in data:
        rewards = item['rewards']

        # Skip failed scenarios
        if not isinstance(rewards[0], (list, tuple)):
            continue

        # Only look at agent2 (index 1 - qwen-2.5-7b)
        if len(rewards) < 2:
            continue

        overall_score, metrics = rewards[1]

        env = item['environment']
        meta = item.get('meta', {})

        # Get full conversation
        messages = meta.get('messages', [[]])[1] if len(meta.get('messages', [])) > 1 else []

        example = {
            'benchmark': benchmark_name,
            'scenario_id': env['id'],
            'agent_name': env['agent_names'][1],
            'overall_score': overall_score,
            'believability': metrics['believability'],
            'relationship': metrics['relationship'],
            'knowledge': metrics['knowledge'],
            'secret': metrics['secret'],
            'social_rules': metrics['social_rules'],
            'financial_and_material_benefits': metrics['financial_and_material_benefits'],
            'goal': metrics['goal'],
            'full_messages': messages
        }
        examples.append(example)

    return pd.DataFrame(examples)

# Load both datasets
print("Loading data...\n")
df_all = load_and_analyze('../results/sotopia_all_gpt-4o_qwen-2.5-7b-instruct.jsonl', 'Sotopia All')
df_hard = load_and_analyze('../results/sotopia_hard_gpt-4o_qwen-2.5-7b-instruct_1.jsonl', 'Sotopia Hard')

df = pd.concat([df_all, df_hard], ignore_index=True)

print("="*80)
print("LOW SCORE EXAMPLES BY METRIC")
print("="*80)

metrics = ['believability', 'relationship', 'knowledge', 'secret',
           'social_rules', 'financial_and_material_benefits', 'goal']

for metric in metrics:
    print(f"\n{'='*80}")
    print(f"LOWEST {metric.upper().replace('_', ' ')} EXAMPLES")
    print(f"{'='*80}\n")

    # Get bottom 3 examples for this metric
    bottom_examples = df.nsmallest(3, metric)

    for idx, row in bottom_examples.iterrows():
        print(f"Example {idx+1}:")
        print(f"  Benchmark: {row['benchmark']}")
        print(f"  Agent: {row['agent_name']}")
        print(f"  Overall Score: {row['overall_score']:.3f}")
        print(f"\n  Scores:")
        print(f"    {metric}: {row[metric]}")
        print(f"    believability: {row['believability']}")
        print(f"    relationship: {row['relationship']}")
        print(f"    knowledge: {row['knowledge']}")
        print(f"    goal: {row['goal']}")
        print(f"    financial_benefits: {row['financial_and_material_benefits']}")
        print(f"    secret: {row['secret']}")
        print(f"    social_rules: {row['social_rules']}")

        print(f"\n  FULL CONVERSATION:")
        print("  " + "-"*76)
        for msg in row['full_messages']:
            if len(msg) >= 3:
                speaker = msg[0]
                recipient = msg[1]
                content = msg[2]
                print(f"\n  {speaker} -> {recipient}:")
                print(f"    {content}")
        print("  " + "-"*76)
        print()

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
