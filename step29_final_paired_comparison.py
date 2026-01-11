import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
ASSISTED_FILE = "llm_assisted_logistic_top5_model_results.csv"
UNASSISTED_FILE = "cluster_classification_results_logistic_llm_unassisted.csv"
OUTPUT_DIR = "comparison_plots_logistic"
FONT_SIZE = 18

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- LOAD & MERGE ---
df_as = pd.read_csv(ASSISTED_FILE)
df_un = pd.read_csv(UNASSISTED_FILE)

# Get max AUC per dataset for both methods
df_as_sub = df_as.groupby("Dataset")["AUC"].max().reset_index().rename(columns={"AUC": "AUC_assisted"})
df_un_sub = df_un.groupby("Dataset")["AUC"].max().reset_index().rename(columns={"AUC": "AUC_unassisted"})

df_merged = pd.merge(df_un_sub, df_as_sub, on="Dataset")
df_merged = df_merged[(df_merged["AUC_assisted"] >= 0.6) & (df_merged["AUC_unassisted"] >= 0.6)]
df_merged["LLM_Better"] = df_merged["AUC_assisted"] > df_merged["AUC_unassisted"]

def mean_ci(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
    return mean, margin

def plot_comparison(df_sub, suffix, dpi=300):
    plt.figure(figsize=(10, 7), dpi=dpi)
    
    # Paired Lines
    for i in range(len(df_sub)):
        plt.plot(["Unassisted", "LLM-Assisted"], 
                 [df_sub.iloc[i]["AUC_unassisted"], df_sub.iloc[i]["AUC_assisted"]],
                 linestyle="dashed", color="gray", alpha=0.3)

    # Distribution markers
    sns.stripplot(x=["Unassisted"]*len(df_sub), y=df_sub["AUC_unassisted"], color="blue", alpha=0.3, jitter=True)
    sns.stripplot(x=["LLM-Assisted"]*len(df_sub), y=df_sub["AUC_assisted"], color="red", alpha=0.3, jitter=True)

    # Mean + CI
    m_un, ci_un = mean_ci(df_sub["AUC_unassisted"])
    m_as, ci_as = mean_ci(df_sub["AUC_assisted"])

    plt.errorbar("Unassisted", m_un, yerr=ci_un, fmt='o', color="blue", capsize=5, elinewidth=3, label="Unassisted Mean")
    plt.errorbar("LLM-Assisted", m_as, yerr=ci_as, fmt='o', color="red", capsize=5, elinewidth=3, label="Assisted Mean")

    # Stats
    _, p_val = stats.ttest_rel(df_sub["AUC_unassisted"], df_sub["AUC_assisted"])
    plt.title(f"Comparison: {suffix}\nPaired t-test p = {p_val:.4f}", fontsize=FONT_SIZE)
    plt.ylabel("AUC Score", fontsize=FONT_SIZE)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"AUC_Compare_{suffix}.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    if not df_merged.empty:
        plot_comparison(df_merged, "All_Datasets", dpi=300)
        df_better = df_merged[df_merged["LLM_Better"]]
        if not df_better.empty:
            plot_comparison(df_better, "LLM_Better_Only", dpi=600)
        print(f"✅ Final Analysis Complete. N={len(df_merged)} datasets.")
