import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ✅ Load the results file
RESULTS_FILE = "cluster_similarity_results.csv"

if not os.path.exists(RESULTS_FILE):
    print(f"❌ ERROR: {RESULTS_FILE} not found. Run Step 7 first!")
else:
    results_df = pd.read_csv(RESULTS_FILE)

    # ✅ Formatting
    if "Dataset" in results_df.columns:
        results_df["Function Type"] = results_df["Dataset"].str.extract(r'_(linear|exponential|quadratic|cubic)')
    else:
        results_df["Function Type"] = "Unknown"

    sns.set(style="whitegrid")

    # 📊 Boxplot for Distance Correlation
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="Function Type", y="Distance Correlation", hue="Model", data=results_df, palette="viridis")
    plt.title("Distribution of Distance Correlation by Function and Model", fontsize=18)
    plt.legend(title="Model", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("distribution_distance_correlation.png")
    print("📊 Saved: distribution_distance_correlation.png")

    # 📊 Boxplot for Cosine Similarity
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="Function Type", y="Cosine Similarity", hue="Model", data=results_df, palette="coolwarm")
    plt.title("Distribution of Cosine Similarity by Function and Model", fontsize=18)
    plt.legend(title="Model", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("distribution_cosine_similarity.png")
    print("📊 Saved: distribution_cosine_similarity.png")

    # 📊 Detailed Boxplots per Function Type
    function_types = results_df["Function Type"].dropna().unique()
    for ftype in function_types:
        plt.figure(figsize=(12, 6))
        subset = results_df[results_df["Function Type"] == ftype]
        sns.boxplot(x="Model", y="Cosine Similarity", data=subset, palette="coolwarm")
        plt.title(f"Detailed Cosine Similarity: {ftype.capitalize()}", fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        fname = f"distribution_detail_{ftype}.png"
        plt.savefig(fname)
        print(f"📊 Saved detailed plot: {fname}")
        plt.close()
