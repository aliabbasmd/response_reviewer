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

# Max AUC per dataset to compare best-case scenarios
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
    
    # Paired Trajectories
    for i in range(len(df_sub)):
        plt.plot(["Unassisted", "LLM-Assisted"], 
                 [df_sub.iloc[i]["AUC_unassisted"], df_sub.iloc[i]["AUC_assisted"]],
                 linestyle="dashed", color="gray", alpha=0.3)

    # Stripplots for Raw Distribution
    sns.stripplot(x=["Unassisted"]*len(df_sub), y=df_sub["AUC_unassisted"], color="blue", alpha=0.3, jitter=True)
    sns.stripplot(x=["LLM-Assisted"]*len(df_sub), y=df_sub["AUC_assisted"], color="red", alpha=0.3, jitter=True)

    # Calculate Stats
    m_un, ci_un = mean_ci(df_sub["AUC_unassisted"])
    m_as, ci_as = mean_ci(df_sub["AUC_assisted"])

    # Error Bars
    plt.errorbar("Unassisted", m_un, yerr=ci_un, fmt='o', color="blue", capsize=5, elinewidth=3, label="Unassisted Mean ± 95% CI")
    plt.errorbar("LLM-Assisted", m_as, yerr=ci_as, fmt='o', color="red", capsize=5, elinewidth=3, label="Assisted Mean ± 95% CI")

    # Boxplot overlay for quartiles
    sns.boxplot(data=df_sub[["AUC_unassisted", "AUC_assisted"]], orient="v", showfliers=False, boxprops=dict(alpha=0.2))

    # T-Test Calculation
    _, p_val = stats.ttest_rel(df_sub["AUC_unassisted"], df_sub["AUC_assisted"])
    plt.title(f"Empirical Evaluation: {suffix.replace('_', ' ')}\nN = {len(df_sub)}", fontsize=FONT_SIZE)
    plt.ylabel("AUC Score", fontsize=FONT_SIZE)
    
    # Add p-value below plot
    plt.gcf().text(0.5, 0.02, f"Paired t-test p = {p_val:.4f}", fontsize=FONT_SIZE, color="red", ha="center", weight="bold")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    
    save_name = f"Final_Validation_{suffix}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), bbox_inches="tight")
    print(f"📈 Saved: {save_name} at {dpi} DPI")
    plt.close()

if __name__ == "__main__":
    if not df_merged.empty:
        # All Datasets
        plot_comparison(df_merged, "All_Datasets", dpi=300)
        
        # High Res for LLM Better
        df_better = df_merged[df_merged["LLM_Better"]]
        if not df_better.empty:
            plot_comparison(df_better, "LLM_Superior_Cases", dpi=600)
            
        print(f"\n✅ Statistical validation complete. Final N: {len(df_merged)}")
