import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# **File Paths**
DATA_FILE = "best_cluster_per_dataset.csv"
OUTPUT_PLOTS_DIR = "comparison_plots"

# Ensure output directory exists
if not os.path.exists(OUTPUT_PLOTS_DIR):
    os.makedirs(OUTPUT_PLOTS_DIR)

# **Load Dataset**
if not os.path.exists(DATA_FILE):
    print(f"❌ ERROR: {DATA_FILE} not found. Ensure Step 25 ran successfully.")
else:
    df = pd.read_csv(DATA_FILE)

    # **Clean Data**
    df["AUC"] = pd.to_numeric(df["AUC"], errors="coerce")
    df = df.dropna(subset=["AUC"])

    # **Mapping Labels**
    cluster_labels = {
        "cluster_e5_small": "E5",
        "cluster_llama": "LLaMA 2 30B",
        "cluster_minilm": "MiniLLM",
        "cluster_ernie": "Ernie",
        "cluster_gatortron": "GatorTron",
        "cluster_roberta": "RoBERTa",
        "cluster_bert": "BERT",
        "cluster_t5": "T5"
    }

    df["Cluster_Label"] = df["Cluster"].map(cluster_labels)

    # **Aggregate Stats**
    cluster_stats = df.groupby(["Cluster_Label", "BestModel"])["AUC"].agg(["mean", "sem"]).reset_index()
    cluster_stats = cluster_stats.sort_values(by="mean", ascending=False)

    # **Significance Testing**
    significant_clusters = []
    for cluster in cluster_stats["Cluster_Label"].unique():
        cluster_data = df[df["Cluster_Label"] == cluster]["AUC"]
        other_data = df[df["Cluster_Label"] != cluster]["AUC"]
        _, p_val = stats.ttest_ind(cluster_data, other_data, equal_var=False, nan_policy='omit')
        if p_val < 0.05:
            significant_clusters.append(cluster)

    # **Plotting Configuration**
    model_palette = {"RandomForest": "blue", "GradientBoost": "green", "LogisticReg": "orange", "NA": "gray"}
    x_labels = cluster_stats["Cluster_Label"].unique()
    x_positions = np.arange(len(x_labels))

    plt.figure(figsize=(14, 7), dpi=600)

    # **Plot Means & SEM**
    for idx, row in cluster_stats.iterrows():
        plt.errorbar(row["Cluster_Label"], row["mean"], yerr=row["sem"], fmt='o', 
                     color=model_palette.get(row["BestModel"], "black"), capsize=5, 
                     markersize=10, elinewidth=2, 
                     label=row["BestModel"] if row["BestModel"] not in plt.gca().get_legend_handles_labels()[1] else "")

    # **Red Star Significance Indicators**
    for cluster in significant_clusters:
        if cluster in x_labels:
            pos = np.where(x_labels == cluster)[0][0]
            val = cluster_stats.loc[cluster_stats["Cluster_Label"] == cluster, "mean"].values[0]
            plt.text(pos, val + 0.02, "*", ha='center', va='bottom', fontsize=25, color="red")

    # **Final Formatting**
    plt.xlabel("LLM Cluster Source", fontsize=16)
    plt.ylabel("Mean Classification AUC", fontsize=16)
    plt.title("Statistical Validation of LLM Clusters", fontsize=18, fontweight="bold")
    plt.xticks(x_positions, x_labels, rotation=45, ha="right", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Handle Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, title="Best Predictive Model", fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))

    plt.figtext(0.15, -0.05, "* Red stars indicate clusters with statistically significant AUC differences (p < 0.05) vs other models.", 
                fontsize=12, color="red", ha="left")

    plt.tight_layout()

    # **Save Exports**
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, "auc_significance_report.jpg"), format="jpeg", dpi=600)
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, "auc_significance_report.png"), format="png", dpi=600)
    
    print(f"✅ Final AUC significance reports generated in /{OUTPUT_PLOTS_DIR}")
