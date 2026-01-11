import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
import os

# ✅ Load the results CSV file
RESULTS_FILE = "cluster_similarity_results.csv"

if not os.path.exists(RESULTS_FILE):
    print(f"❌ ERROR: {RESULTS_FILE} not found. Run Step 7 first!")
else:
    df = pd.read_csv(RESULTS_FILE)

    # ✅ Compute stats
    stats_df = df.groupby(["Function Type", "Model"])[["Distance Correlation", "Cosine Similarity"]].agg(["mean", sem]).reset_index()
    stats_df.columns = ["Function Type", "Model", "Distance Mean", "Distance SEM", "Cosine Mean", "Cosine SEM"]

    sns.set(style="whitegrid")
    function_types = stats_df["Function Type"].unique()

    for func_type in function_types:
        for metric, metric_name, y_min in zip(
            ["Distance Mean", "Cosine Mean"], 
            ["Distance Correlation", "Cosine Similarity"], 
            [None, 0.62]
        ):
            subset = stats_df[stats_df["Function Type"] == func_type].sort_values(by=metric, ascending=False)
            sem_column = "Distance SEM" if metric == "Distance Mean" else "Cosine SEM"

            plt.figure(figsize=(12, 8))
            
            # Error bars and Scatter
            plt.errorbar(subset["Model"], subset[metric], yerr=subset[sem_column], fmt="none", capsize=5, color="black", zorder=2)
            plt.scatter(subset["Model"], subset[metric], color="red", s=100, label="Mean", zorder=3)

            # Annotations
            for i, row in subset.iterrows():
                plt.text(row["Model"], row[metric] - row[sem_column] - 0.02, 
                         f"{row[metric]:.2f} ± {row[sem_column]:.2f}", 
                         ha="center", fontsize=10, fontweight="bold")

            plt.xlabel("Model", fontsize=14)
            plt.ylabel(f"{metric_name} Mean", fontsize=14)
            plt.title(f"{metric_name} Analysis: {func_type.upper()}", fontsize=16)
            
            if y_min is not None:
                plt.ylim(y_min, 1.0)

            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the figure
            filename = f"report_{func_type}_{metric_name.replace(' ', '_')}.png"
            plt.savefig(filename)
            print(f"📊 Report saved: {filename}")
            plt.close()
