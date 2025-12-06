import pandas as pd
import os

# Define metrics and input file
INPUT_FILE = "analysis/sae/sae_social_intel_correlations.csv"
OUTPUT_FILE = "features_to_analyze.txt"
METRICS = [
    "believability",
    "financial_and_material_benefits",
    "goal",
    "knowledge",
    "relationship"
]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    unique_features = set()
    
    for metric in METRICS:
        # Filter for the specific metric
        metric_df = df[df['dimension'] == metric].copy()
        
        # Sort by absolute correlation
        metric_df['abs_corr'] = metric_df['pearson_corr'].abs()
        top_20 = metric_df.sort_values(by='abs_corr', ascending=False).head(20)
        
        # Add to set
        features = top_20['feature_index'].tolist()
        unique_features.update(features)
        
        print(f"Top 20 for {metric}: {features}")

    # Convert to sorted list
    sorted_features = sorted(list(unique_features))
    
    print(f"\nTotal unique features: {len(sorted_features)}")
    
    # Save to file
    with open(OUTPUT_FILE, "w") as f:
        for feat in sorted_features:
            f.write(f"{feat}\n")
            
    print(f"Saved feature list to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
