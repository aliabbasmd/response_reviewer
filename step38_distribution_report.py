import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
UNASSISTED_FILE = "unassisted_linear_model_results.csv"
ASSISTED_FILE = "llm_assisted_regression_results/llm_assisted_regression_results.csv"
OUTPUT_PLOT = "distribution_comparison_report.png"

# --- LOAD & PREP ---
if not os.path.exists(UNASSISTED_FILE) or not os.path.exists(ASSISTED_FILE):
    print("❌ Error: Missing result files. Ensure Steps 17 and 31 ran successfully.")
else:
    df_un = pd.read_csv(UNASSISTED_FILE)
    df_as = pd.read_csv(ASSISTED_FILE)

    merged_df = pd.merge(df_un[['Dataset', 'R2']], df_as[['Dataset', 'R2']], 
                         on='Dataset', suffixes=('_unassisted', '_llm')).dropna()

    # Calculate Metrics
    win_rate = (merged_df["R2_llm"] > merged_df["R2_unassisted"]).mean() * 100
    t_stat, p_val = stats.ttest_rel(merged_df["R2_unassisted"], merged_df["R2_llm"])

    def get_ci(data):
        m = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
        return m, h

    m_un, h_un = get_ci(merged_df["R2_unassisted"])
    m_llm, h_llm = get_ci(merged_df["R2_llm"])

    # --- PLOTTING ---
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Layer 1: Boxplots (The background)
    melted = merged_df.melt(id_vars=["Dataset"], var_name="Method", value_name="R2")
    melted["Method"] = melted["Method"].map({"R2_unassisted": "Unassisted", "R2_llm": "LLM-Assisted"})
    sns.boxplot(data=melted, x="Method", y="R2", width=0.4, palette=["#3498db", "#e74c3c"], 
                showfliers=False, boxprops=dict(alpha=0.2))

    # Layer 2: Jittered Points (The raw data)
    def jitter(val, n): return val + np.random.uniform(-0.1, 0.1, n)
    plt.scatter(jitter(np.zeros(len(merged_df)), len(merged_df)), merged_df["R2_unassisted"], 
                color="blue", alpha=0.15, s=25, label="Raw Data (Unassisted)")
    plt.scatter(jitter(np.ones(len(merged_df)), len(merged_df)), merged_df["R2_llm"], 
                color="red", alpha=0.15, s=25, label="Raw Data (Assisted)")

    # Layer 3: Error Bars (The statistical summary)
    plt.errorbar(0, m_un, yerr=h_un, fmt='o', color="darkblue", capsize=6, markersize=10, elinewidth=3, label="Mean ± 95% CI")
    plt.errorbar(1, m_llm, yerr=h_llm, fmt='o', color="darkred", capsize=6, markersize=10, elinewidth=3)

    # Annotations
    plt.title(f"Performance Distribution & Statistical Delta\nLLM Win Rate: {win_rate:.1f}%", fontsize=16, fontweight="bold")
    plt.ylabel("R² Score (Goodness of Fit)", fontsize=14)
    
    # Dynamically place p-value text
    y_max = max(merged_df["R2_unassisted"].max(), merged_df["R2_llm"].max())
    if p_val < 0.05:
        plt.text(0.5, y_max, f"Paired t-test: p = {p_val:.4f}*", 
                 ha="center", fontsize=12, color="red", fontweight="bold")
    else:
        plt.text(0.5, y_max, f"Paired t-test: p = {p_val:.4f} (NS)", 
                 ha="center", fontsize=12, color="black")

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    plt.tight_layout()
    
    # Save and cleanup
    plt.savefig(OUTPUT_PLOT, bbox_inches="tight")
    plt.close()
    print(f"📊 Final distribution report saved to {OUTPUT_PLOT}")
