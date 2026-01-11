import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load the results file
RESULTS_FILE = "cluster_similarity_results.csv"

if not os.path.exists(RESULTS_FILE):
    print(f"❌ ERROR: {RESULTS_FILE} not found. Run Step 7 first!")
else:
    results_df = pd.read_csv(RESULTS_FILE)

    # ✅ Extract the function type
    results_df["Function Type"] = results_df["Dataset"].str.extract(r'_(linear|exponential|quadratic|cubic)')

    # Set visualization style
    sns.set(style="whitegrid")

    # 📊 Boxplot for Distance Correlation Grouped by Function Type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Function Type", y="Distance Correlation", data=results_df, palette="viridis")
    plt.xlabel("Function Type", fontsize=16)
    plt.ylabel("Distance Correlation", fontsize=16)
    plt.title("Ease of Interpretation: Distance Correlation by Function Type", fontsize=18)
    
    plt.tight_layout()
    plt.savefig("summary_function_distance.png")
    print("📊 Saved: summary_function_distance.png")

    # 📊 Boxplot for Cosine Similarity Grouped by Function Type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Function Type", y="Cosine Similarity", data=results_df, palette="coolwarm")
    plt.xlabel("Function Type", fontsize=16)
    plt.ylabel("Cosine Similarity", fontsize=16)
    plt.title("Structural Fidelity: Cosine Similarity by Function Type", fontsize=18)
    
    plt.tight_layout()
    plt.savefig("summary_function_cosine.png")
    print("📊 Saved: summary_function_cosine.png")
    plt.close()
