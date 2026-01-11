import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
ASSISTED_FILE = "llm_assisted_linear_results_with_metrics/llm_assisted_regression_results.csv"
UNASSISTED_FILE = "numeric_x3_results_extended/unassisted_model_results_extended.csv"
OUTPUT_DIR = "comparison_plots"
FONT_SIZE = 18

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- LOAD DATA ---
df_as = pd.read_csv(ASSISTED_FILE)
df_un = pd.read_csv(UNASSISTED_FILE)

# Filter for models that learned a signal (R2 > 0.2)
df_as = df_as[df_as["R2"] > 0.2]
df_un = df_un[df_un["R2"] > 0.2]

# Merge on Dataset for paired comparison
df_merged = pd.merge(df_un, df_as, on="Dataset", suffixes=("_unassisted", "_assisted"))

# Impact Classification
df_merged["LLM_Better"] = df_merged["MAE_assisted"] < df_merged["MAE_unassisted"]
groups = {
    "LLM_Assisted_Superior": df_merged[df_merged["LLM_Better"]],
    "LLM_Assisted_Inferior": df_merged[~df_merged["LLM_Better"]]
}

def mean_ci(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
    return mean, margin

def plot_mae_comparison(df_sub, title_suffix, file_suffix):
    if df_sub.empty: return
    
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Paired slope lines
    for i in range(len(df_sub)):
        plt.plot(["Unassisted", "LLM-Assisted"], 
                 [df_sub.iloc[i]["MAE_unassisted"], df_sub.iloc[i]["MAE_assisted"]],
                 linestyle="dashed", color="gray", alpha=0.3)

    # Stripplots for individual data points
    sns.stripplot(x=["Unassisted"]*len(df_sub), y=df_sub["MAE_unassisted"], color="blue", alpha=0.3, jitter=True)
    sns.stripplot(x=["LLM-Assisted"]*len(df_sub), y=df_sub["MAE_assisted"], color="red", alpha=0.3, jitter=True)

    # Stats
    m_un, ci_un = mean_ci(df_sub["MAE_unassisted"])
    m_as, ci_as = mean_ci(df_sub["MAE_assisted"])

    # Error Bars
    plt.errorbar("Unassisted", m_un, yerr=ci_un, fmt='o', color="blue", capsize=5, elinewidth=3, label="Unassisted Mean ± 95% CI")
    plt.errorbar("LLM-Assisted", m_as, yerr=ci_as, fmt='o', color="red", capsize=5, elinewidth=3, label="Assisted Mean ± 95% CI")

    # Boxplot for quartiles
    sns.boxplot(data=df_sub[["MAE_unassisted", "MAE_assisted"]], orient="v", showfliers=False, boxprops=dict(alpha=0.2))

    # Significance
    t_stat, p_val = stats.ttest_rel(df_sub["MAE_unassisted"], df_sub["MAE_assisted"])
    
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=16)
    plt.title(f"MAE Analysis: {title_suffix}\nN = {len(df_sub)} | p = {p_val:.4f}", fontsize=18, fontweight="bold")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"MAE_Segmented_{file_suffix}.png"))
    print(f"📊 Saved segmented analysis: MAE_Segmented_{file_suffix}.png")
    plt.close()

if __name__ == "__main__":
    for name, subset in groups.items():
        plot_mae_comparison(subset, name.replace("_", " "), name)
