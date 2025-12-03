"""
Explore Sotopia dataset to find interesting scenarios for steering tests
"""

import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/demiw/anlp-fall2025-final-project/results/sotopia_all_gpt-4o_Qwen_Qwen2.5-7B-Instruct_20251201_merged.jsonl",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all records (default: show first 20)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Sotopia Dataset Explorer")
    print("=" * 80)

    records = []
    with open(args.data_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    print(f"\nTotal records: {len(records)}")
    print("\n" + "=" * 80)

    limit = len(records) if args.show_all else 20

    for i, record in enumerate(records[:limit]):
        # Extract scenario
        first_turn = record['meta']['messages'][0]
        context = first_turn[0][2]

        scenario_start = context.find("Scenario:") + len("Scenario:")
        scenario_end = context.find("Participants:")
        scenario = context[scenario_start:scenario_end].strip()

        # Extract agent names
        agents = record['environment']['agent_names']

        # Extract scores
        rewards = record['rewards']
        if len(rewards) > 0:
            scores = rewards[0][1]  # First agent's scores
            believability = scores.get('believability', 0)
            relationship = scores.get('relationship', 0)
            knowledge = scores.get('knowledge', 0)
            goal = scores.get('goal', 0)
            financial = scores.get('financial_and_material_benefits', 0)
            overall = scores.get('overall_score', 0)

        print(f"\n{'='*80}")
        print(f"Record {i}")
        print(f"{'='*80}")
        print(f"Agents: {', '.join(agents)}")
        print(f"\nScenario:")
        print(f"  {scenario[:200]}{'...' if len(scenario) > 200 else ''}")
        print(f"\nScores:")
        print(f"  Overall: {overall:.2f}")
        print(f"  Believability: {believability:.1f}")
        print(f"  Relationship: {relationship:.1f}")
        print(f"  Knowledge: {knowledge:.1f}")
        print(f"  Goal: {goal:.1f}")
        print(f"  Financial: {financial:.1f}")

        # Show first utterance
        if len(record['meta']['messages']) > 1:
            turn_msg = record['meta']['messages'][1]
            if len(turn_msg) > 0:
                first_utterance = turn_msg[0][2]
                if "Turn #0:" in first_utterance:
                    utterance_text = first_utterance.split("Turn #0:")[1].strip()[:150]
                    print(f"\nFirst utterance:")
                    print(f"  {utterance_text}...")

        print(f"\nTo test this scenario, use:")
        print(f"  python inferences/steering_with_sotopia.py --record-idx {i} --feature-idx <FEATURE> --steering-strength <STRENGTH>")

    if not args.show_all and len(records) > 20:
        print(f"\n{'='*80}")
        print(f"Showing first 20 of {len(records)} records.")
        print(f"Use --show-all to see all records.")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
