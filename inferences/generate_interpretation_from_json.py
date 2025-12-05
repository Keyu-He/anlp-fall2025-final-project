import json
import argparse
import os
from openai import OpenAI

def interpret_results(feature_idx, all_results):
    """
    Uses an LLM to interpret the feature based on generation differences across multiple scenarios.
    Returns the interpretation string.
    """
    try:
        client = OpenAI()
    except Exception as e:
        print(f"Could not initialize OpenAI client: {e}. Skipping automated interpretation.")
        return None

    prompt = f"""
You are an expert in interpreting Sparse Autoencoder (SAE) features in Language Models.
I will provide you with text generations from a model across multiple scenarios.
For each scenario, I will show:
1. Negative Steering: The feature was suppressed or steered negatively.
2. Baseline: The original model behavior.
3. Positive Steering: The feature was activated or steered positively.

Your task is to analyze the differences across ALL scenarios and find the COMMON underlying semantic meaning of SAE Feature {feature_idx}.
Do NOT focus on the specific topic of one scenario (e.g., if one scenario is about trash, don't just say "trash"). Look for the abstract behavior or concept that explains the changes in all scenarios.

"""

    for scenario, results in all_results.items():
        strengths = sorted(results.keys(), key=lambda x: float(x))
        if len(strengths) < 3:
            continue
            
        neg_strength = strengths[0]
        baseline_strength = "0.0" if "0.0" in results else strengths[len(strengths)//2]
        pos_strength = strengths[-1]
        
        prompt += f"""
### Scenario: {scenario}
[Negative Steering (Strength {neg_strength})]
{results[neg_strength]}

[Baseline (Strength {baseline_strength})]
{results[baseline_strength]}

[Positive Steering (Strength {pos_strength})]
{results[pos_strength]}
"""

    prompt += f"""
Based on these outputs, what specific concept, behavior, or topic does Feature {feature_idx} represent?
Provide a concise title (3-5 words) and a brief explanation that fits ALL scenarios.
"""
    
    print(f"\n{'='*40}")
    print("AUTOMATED INTERPRETATION (GPT-4o)")
    print(f"{'='*40}")
    
    interpretation = None
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful interpretability assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        interpretation = response.choices[0].message.content
        print(interpretation)
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"Error during API call: {e}")
        print("Prompt that would have been sent:")
        print(prompt)
        
    return interpretation

def main():
    parser = argparse.ArgumentParser(description="Generate interpretation from JSON results")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing results")
    args = parser.parse_args()
    
    with open(args.json_file, "r") as f:
        data = json.load(f)
        
    feature_idx = data["feature_idx"]
    scenarios = data["scenarios"]
    
    interpretation = interpret_results(feature_idx, scenarios)
    
    if interpretation:
        data["interpretation"] = interpretation
        with open(args.json_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Updated interpretation saved to {args.json_file}")

if __name__ == "__main__":
    main()
