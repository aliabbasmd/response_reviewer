import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# --- CONFIG ---
UNASSISTED_FILE = "unassisted_linear_model_results.csv"
ASSISTED_FILE = "llm_assisted_regression_results/llm_assisted_regression_results.csv"
OUTPUT_PLOT = "regression_comparison_stats.png"

# --- LOAD DATA ---
if not os.path.exists(UNASSISTED_FILE) or not os.path.exists(ASSISTED_FILE):
    print("❌ Error: Missing result files. Ensure Steps 17 and 31 ran successfully.")
else:
    df_un = pd.read_csv(UNASSISTED_FILE)
    df_as = pd.read_csv(ASSISTED_FILE)

    # Merge on Dataset
    merged_df = pd.merge(
        df_un[['Dataset', 'R2']], 
        df_as[['Dataset', 'R2']], 
        on='Dataset', 
        suffixes=('_unassisted', '_llm')
    ).dropna()

    def mean_ci(data, confidence=0.95):
        mean_val = np.mean(data)
        sem = stats.sem(data)
        margin = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
        return mean_val, margin

    # Calculate Stats
    m_un, ci_un = mean_ci(merged_df["R2_unassisted"])
    m_llm, ci_llm = mean_ci(merged_df["R2_llm"])

    # Statistical Testing
    t_stat, p_ttest = stats.ttest_rel(merged_df["R2_unassisted"], merged_df["R2_llm"])
    w_stat, p_wilcoxon = stats.wilcoxon(merged_df["R2_unassisted"], merged_df["R2_llm"])

    print(f"\n--- Statistical Comparison (N={len(merged_df)}) ---")
    print(f"Paired t-test: p = {p_ttest:.4f}")
    print(f"Wilcoxon Test: p = {p_wilcoxon:.4f}")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 7), dpi=300)
    methods = ["Unassisted", "LLM-Assisted"]
    means = [m_un, m_llm]
    yerrs = [ci_un, ci_llm]
    colors = ["#3498db", "#e74c3c"]

    for i, method in enumerate(methods):
        plt.errorbar(
            method, means[i], yerr=yerrs[i], 
            fmt='o', capsize=8, capthick=2, markersize=10, 
            elinewidth=3, color=colors[i], label=f"{method} (Mean ± 95% CI)"
        )

    # Highlight Significance
    if p_ttest < 0.05:
        plt.text(0.5, max(means) + 0.05, f"* p < 0.05 (Significant Improvement)", 
                 ha="center", fontsize=12, color="red", fontweight="bold")

    plt.ylabel("Mean R² Score", fontsize=14)
    plt.title("Statistical Comparison: Unassisted vs. LLM-Assisted Regression", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim(0, 1.0)
    
    plt.savefig(OUTPUT_PLOT, bbox_inches="tight")
    print(f"📊 Comparative plot saved to {OUTPUT_PLOT}")
