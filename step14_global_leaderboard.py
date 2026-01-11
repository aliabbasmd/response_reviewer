import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ Define the input file (Adjusted to match your Step 5/7 outputs)
SUMMARY_FILE = 'model_summary_results.csv'

if not os.path.exists(SUMMARY_FILE):
    print(f"❌ ERROR: {SUMMARY_FILE} not found. Ensure you have run the summary steps!")
else:
    # Load and clean CSV - matching the structure of your previous summary outputs
    summary_df = pd.read_csv(SUMMARY_FILE)
    
    # Standardize column names if they have MultiIndex headers from previous steps
    if 'Model' not in summary_df.columns:
        summary_df.columns = ['Model', 'Distance Correlation Mean', 'Distance Correlation Std', 
                              'Cosine Similarity Mean', 'Cosine Similarity Std']

    # Convert columns to numeric
    numeric_cols = summary_df.columns[1:]
    summary_df[numeric_cols] = summary_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    sns.set(style="whitegrid")

    # 📊 **Distance Correlation Leaderboard**
    sorted_distance_df = summary_df.sort_values(by="Distance Correlation Mean", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.scatter(sorted_distance_df["Model"], sorted_distance_df["Distance Correlation Mean"], 
                color='red', s=120, edgecolors='black', label="Mean", zorder=3)
    plt.errorbar(sorted_distance_df["Model"], sorted_distance_df["Distance Correlation Mean"], 
                 yerr=sorted_distance_df["Distance Correlation Std"], fmt='none', 
                 ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5, zorder=2)
    plt.xticks(rotation=45, fontsize=12)
    plt.ylabel("Mean Distance Correlation", fontsize=14)
    plt.title("Model Leaderboard: Structural Alignment (Sorted)", fontsize=16)
    plt.tight_layout()
    plt.savefig("leaderboard_distance.png")
    print("📊 Saved: leaderboard_distance.png")

    # 📊 **Cosine Similarity Leaderboard**
    sorted_cosine_df = summary_df.sort_values(by="Cosine Similarity Mean", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.scatter(sorted_cosine_df["Model"], sorted_cosine_df["Cosine Similarity Mean"], 
                color='blue', s=120, edgecolors='black', label="Mean", zorder=3)
    plt.errorbar(sorted_cosine_df["Model"], sorted_cosine_df["Cosine Similarity Mean"], 
                 yerr=sorted_cosine_df["Cosine Similarity Std"], fmt='none', 
                 ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5, zorder=2)
    plt.xticks(rotation=45, fontsize=12)
    plt.ylabel("Mean Cosine Similarity", fontsize=14)
    plt.title("Model Leaderboard: Semantic Fidelity (Sorted)", fontsize=16)
    plt.tight_layout()
    plt.savefig("leaderboard_cosine.png")
    print("📊 Saved: leaderboard_cosine.png")
