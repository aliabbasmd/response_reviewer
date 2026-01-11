import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
UNASSISTED_FILE = "unassisted_linear_model_results.csv"
ASSISTED_FILE = "llm_assisted_regression_results/llm_assisted_regression_results.csv"
METADATA_FILE = "cluster_similarity_results_augmented.csv"
OUTPUT_TABLE = "llm_performance_comparison_with_CI.csv"
OUTPUT_PLOT = "llm_performance_CI_plot.png"

# --- DATA PREP ---
if not all(os.path.exists(f) for f in [UNASSISTED_FILE, ASSISTED_FILE, METADATA_FILE]):
    print("❌ Error: Missing input files. Ensure previous steps ran successfully.")
else:
    df_un = pd.read_csv(UNASSISTED_FILE)
    df_as = pd.read_csv(ASSISTED_FILE)
    df_meta = pd.read_csv(METADATA_FILE)

    # Merge performance data
    merged = pd.merge(df_un[['Dataset', 'R2']], df_as[['Dataset', 'R2']], 
                      on='Dataset', suffixes=('_unassisted', '_llm'))
    
    # Merge with structural metadata
    merged = pd.merge(merged, df_meta[['Dataset', 'SubjectVariableRatio', 'NumCategoricalVars']], on='Dataset')

    # Filter for valid signal (R2 >= 0.2)
    filtered_df = merged[(merged["R2_unassisted"] >= 0.2) & (merged["R2_llm"] >= 0.2)].copy()
    filtered_df["LLM_Better"] = filtered_df["R2_llm"] > filtered_df["R2_unassisted"]

    def mean_ci(data, confidence=0.95):
        m = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
        return m, m - h, m + h

    results = []
    features = ["SubjectVariableRatio", "NumCategoricalVars"]

    for feature in features:
        group_better = filtered_df[filtered_df["LLM_Better"]][feature]
        group_worse = filtered_df[~filtered_df["LLM_Better"]][feature]

        # Stats
        m_b, low_b, high_b = mean_ci(group_better)
        m_w, low_w, high_w = mean_ci(group_worse)
        t_stat, p_t = stats.ttest_ind(group_better, group_worse, equal_var=False)
        u_stat, p_u = stats.mannwhitneyu(group_better, group_worse)

        results.append({
            "Feature": feature, "Mean_Better": m_b, "CI_Low_B": low_b, "CI_High_B": high_b,
            "Mean_Worse": m_w, "CI_Low_W": low_w, "CI_High_W": high_w,
            "t_p": p_t, "u_p": p_u
        })

    # Save Stats Table
    stats_df = pd.DataFrame(results).round(4)
    stats_df.to_csv(OUTPUT_TABLE, index=False)
    print(f"✅ Statistical table saved to {OUTPUT_TABLE}")

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 6), dpi=300)
    for i, feature in enumerate(features):
        m_b, l_b, h_b = mean_ci(filtered_df[filtered_df["LLM_Better"]][feature])
        m_w, l_w, h_w = mean_ci(filtered_df[~filtered_df["LLM_Better"]][feature])

        plt.errorbar(i - 0.1, m_b, yerr=[[m_b-l_b], [h_b-m_b]], fmt='o', color="#2ecc71", capsize=5, label="LLM Better" if i==0 else "")
        plt.errorbar(i + 0.1, m_w, yerr=[[m_w-l_w], [h_w-m_w]], fmt='o', color="#e74c3c", capsize=5, label="LLM Worse" if i==0 else "")

    plt.xticks([0, 1], features, fontsize=12)
    plt.ylabel("Value Mean ± 95% CI", fontsize=12)
    plt.title("Structural Drivers of LLM Regression Performance", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(OUTPUT_PLOT, bbox_inches="tight")
    print(f"📊 CI Plot saved to {OUTPUT_PLOT}")
