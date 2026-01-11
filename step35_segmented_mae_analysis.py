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

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD & PREP ---
df_as = pd.read_csv(ASSISTED_FILE)
df_un = pd.read_csv(UNASSISTED_FILE)

# Filter for baseline quality
df_as = df_as[df_as["R2"] > 0.2]
df_un = df_un[df_un["R2"] > 0.2]

df_merged = pd.merge(df_un, df_as, on="Dataset", suffixes=("_unassisted", "_assisted"))

# Define Impact Groups
df_merged["LLM_Better"] = df_merged["MAE_assisted"] < df_merged["MAE_unassisted"]
groups = {
    "LLM_Superior": df_merged[df_merged["LLM_Better"]],
    "LLM_Inferior": df_merged[~df_merged["LLM_Better"]]
}

def plot_mae_comparison(df_subset, title_suffix, file_suffix):
    if df_subset.empty: return
    
    plt.figure(figsize=(12, 8))
    
    # Paired Trajectories
    for i in range(len(df_subset)):
        plt.plot(["Unassisted", "LLM-Assisted"], 
                 [df_subset.iloc[i]["MAE_unassisted"], df_subset.iloc[i]["MAE_assisted"]],
                 linestyle="dashed", color="gray", alpha=0.3)

    sns.stripplot(x=["Unassisted"]*len(df_subset), y=df_subset["MAE_unassisted"], jitter=True, alpha=0.3, color="blue")
    sns.stripplot(x=["LLM-Assisted"]*len(df_subset), y=df_subset["MAE_assisted"], jitter=True, alpha=0.3, color="red")

    # Stats Calculation
    def get_ci(data):
        m = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
        return m, h

    m_un, h_un = get_ci(df_subset["MAE_unassisted"])
    m_as, h_as = get_ci(df_subset["MAE_assisted"])

    plt.errorbar("Unassisted", m_un, yerr=h_un, fmt='o', color="blue", capsize=5, elinewidth=3, label="Unassisted Mean ± 95% CI")
    plt.errorbar("LLM-Assisted", m_as, yerr=h_as, fmt='o', color="red", capsize=5, elinewidth=3, label="Assisted Mean ± 95% CI")

    sns.boxplot(data=df_subset[["MAE_unassisted", "MAE_assisted"]], orient="v", showfliers=False, boxprops=dict(alpha=0.2))

    # Significance Test
    t_stat, p_val = stats.ttest_rel(df_subset["MAE_unassisted"], df_subset["MAE_assisted"])
    plt.title(f"MAE Comparison: {title_suffix}\n(N = {len(df_subset)})", fontsize=FONT_SIZE, fontweight="bold")
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=FONT_SIZE)
    
    plt.gcf().text(0.5, 0.02, f"Paired t-test p = {p_val:.4f}", fontsize=FONT_SIZE, color="red", ha="center", weight="bold")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    
    save_path = os.path.join(OUTPUT_DIR, f"MAE_Segmented_{file_suffix}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"📊 Saved Segmented Plot: {save_path}")
    plt.close()

if __name__ == "__main__":
    for name, data in groups.items():
        plot_mae_comparison(data, name.replace("_", " "), name)
