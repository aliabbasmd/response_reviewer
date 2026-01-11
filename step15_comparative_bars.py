import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ Define the input file
SUMMARY_FILE = 'model_summary_results.csv'

if not os.path.exists(SUMMARY_FILE):
    print(f"❌ ERROR: {SUMMARY_FILE} not found. Ensure Step 5/7 has run successfully!")
else:
    # Load and clean CSV
    summary_df = pd.read_csv(SUMMARY_FILE)
    
    # Standardize column names if needed
    if 'Model' not in summary_df.columns:
        summary_df.columns = ['Model', 'Distance Correlation Mean', 'Distance Correlation Std', 
                              'Cosine Similarity Mean', 'Cosine Similarity Std']

    # Convert to numeric
    numeric_cols = summary_df.columns[1:]
    summary_df[numeric_cols] = summary_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    sns.set(style="whitegrid")

    # 📊 **Bar Plot: Distance Correlation**
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Model", y="Distance Correlation Mean", data=summary_df, palette="viridis")

    # Add error bars manually
    for i, bar in enumerate(ax.patches):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        err = summary_df["Distance Correlation Std"].iloc[i]
        plt.errorbar(x, y, yerr=err, fmt='none', ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5)

    plt.xticks(rotation=45, fontsize=12)
    plt.ylabel("Mean Distance Correlation", fontsize=14)
    plt.title("Distance Correlation Magnitude Across Models", fontsize=16)
    plt.tight_layout()
    plt.savefig("bars_distance_correlation.png")
    print("📊 Saved: bars_distance_correlation.png")

    # 📊 **Bar Plot: Cosine Similarity**
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Model", y="Cosine Similarity Mean", data=summary_df, palette="coolwarm")

    # Add error bars manually
    for i, bar in enumerate(ax.patches):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        err = summary_df["Cosine Similarity Std"].iloc[i]
        plt.errorbar(x, y, yerr=err, fmt='none', ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5)

    plt.xticks(rotation=45, fontsize=12)
    plt.ylabel("Mean Cosine Similarity", fontsize=14)
    plt.title("Cosine Similarity Magnitude Across Models", fontsize=16)
    plt.tight_layout()
    plt.savefig("bars_cosine_similarity.png")
    print("📊 Saved: bars_cosine_similarity.png")
