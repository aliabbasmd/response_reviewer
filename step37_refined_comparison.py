import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# --- CONFIG ---
UNASSISTED_FILE = "unassisted_linear_model_results.csv"
ASSISTED_FILE = "llm_assisted_regression_results/llm_assisted_regression_results.csv"
METADATA_FILE = "cluster_similarity_results_augmented.csv"
OUTPUT_PLOT = "refined_comparison_results.png"

# --- DATA PREPARATION ---
if not all(os.path.exists(f) for f in [UNASSISTED_FILE, ASSISTED_FILE, METADATA_FILE]):
    print("❌ Error: Missing input files. Ensure previous steps ran successfully.")
else:
    df_un = pd.read_csv(UNASSISTED_FILE)
    df_as = pd.read_csv(ASSISTED_FILE)
    df_meta = pd.read_csv(METADATA_FILE)

    # Merge performance and metadata
    merged = pd.merge(df_un[['Dataset', 'R2']], df_as[['Dataset', 'R2']], on='Dataset', suffixes=('_unassisted', '_llm'))
    merged = pd.merge(merged, df_meta[['Dataset', 'SubjectVariableRatio', 'NumCategoricalVars']], on='Dataset')

    # Apply Diagnostic Cutoffs
    filtered_df = merged[(merged["SubjectVariableRatio"] >= 20) & (merged["NumCategoricalVars"] <= 25)].copy()

    def mean_ci(data, confidence=0.95):
        mean_val = np.mean(data)
        sem = stats.sem(data)
        margin = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
        return mean_val, margin

    # Compute Statistics
    m_un, ci_un = mean_ci(filtered_df["R2_unassisted"])
    m_llm, ci_llm = mean_ci(filtered_df["R2_llm"])

    t_stat, p_ttest = stats.ttest_rel(filtered_df["R2_unassisted"], filtered_df["R2_llm"])
    w_stat, p_wilcoxon = stats.wilcoxon(filtered_df["R2_unassisted"], filtered_df["R2_llm"])

    print(f"\n--- Refined Analysis Results (N={len(filtered_df)}) ---")
    print(f"Paired t-test: p = {p_ttest:.4f}")
    print(f"Wilcoxon p-value: {p_wilcoxon:.4f}")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 7), dpi=300)
    methods = ["Unassisted", "LLM-Assisted"]
    means = [m_un, m_llm]
    yerrs = [ci_un, ci_llm]
    colors = ["#3498db", "#e74c3c"]

    for i, method in enumerate(methods):
        plt.errorbar(method, means[i], yerr=yerrs[i], fmt='o', capsize=8, 
                     capthick=2, markersize=10, elinewidth=3, color=colors[i])

    if p_ttest < 0.05:
        plt.text(0.5, max(means) + 0.05, f"* Significant (p < 0.05)", 
                 ha="center", fontsize=12, color="red", fontweight="bold")

    plt.ylabel("Mean R² Score", fontsize=14)
    plt.title("Refined Model Comparison\n(Filtered: SVR ≥ 20, Categories ≤ 25)", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim(0, 1.0)
    
    plt.savefig(OUTPUT_PLOT, bbox_inches="tight")
    print(f"📊 Refined comparative plot saved to {OUTPUT_PLOT}")
