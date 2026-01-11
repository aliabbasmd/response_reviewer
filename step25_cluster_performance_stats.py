import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load and clean
df = pd.read_csv("classification_results/cluster_classification_results.csv")
df["AUC"] = pd.to_numeric(df["AUC"], errors="coerce")
df = df.dropna(subset=["AUC"])

# Save Best per Dataset
best_df = df.loc[df.groupby("Dataset")["AUC"].idxmax()]
best_df.to_csv("best_cluster_per_dataset.csv", index=False)

# Stats & Mapping
cluster_labels = {"cluster_e5_small": "E5", "cluster_llama": "LLaMA 2", "cluster_minilm": "MiniLLM", "cluster_ernie": "Ernie", "cluster_gatortron": "GatorTron", "cluster_roberta": "RoBERTa", "cluster_bert": "BERT", "cluster_t5": "T5"}
best_df["Label"] = best_df["Cluster"].map(cluster_labels)

stats_df = best_df.groupby("Label")["AUC"].agg(["mean", "count", "std"]).reset_index()
stats_df["sem"] = stats_df["std"] / np.sqrt(stats_df["count"])
stats_df["ci95"] = stats_df["sem"] * stats.t.ppf(0.975, df=stats_df["count"]-1)
stats_df = stats_df.sort_values(by="mean", ascending=False)

# Plotting
plt.figure(figsize=(14, 7), dpi=300)
plt.errorbar(stats_df["Label"], stats_df["mean"], yerr=stats_df["ci95"], fmt='o', color='black', capsize=5, markersize=8, elinewidth=2)
plt.xlabel("LLM Cluster Source", fontsize=14)
plt.ylabel("Mean Classification AUC", fontsize=14)
plt.title("Predictive Stability of LLM Clusters (95% Confidence Intervals)", fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("llm_predictive_stability.png")
print("📊 Saved: llm_predictive_stability.png")
