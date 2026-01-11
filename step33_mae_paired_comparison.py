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

# Filter for models that actually learned something (R2 > 0.2)
df_as = df_as[df_as["R2"] > 0.2]
df_un = df_un[df_un["R2"] > 0.2]

# Merge on Dataset to ensure paired comparison
df_merged = pd.merge(df_un, df_as, on="Dataset", suffixes=("_unassisted", "_assisted"))

def mean_ci(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
    return mean, margin

# --- PLOTTING ---
plt.figure(figsize=(10, 7), dpi=300)

# Paired Trajectories (Dashed lines showing individual dataset improvement)
for i in range(len(df_merged)):
    plt.plot(["Unassisted", "LLM-Assisted"], 
             [df_merged.iloc[i]["MAE_unassisted"], df_merged.iloc[i]["MAE_assisted"]],
             linestyle="dashed", color="gray", alpha=0.3)

# Distribution Overlays
sns.stripplot(x=["Unassisted"]*len(df_merged), y=df_merged["MAE_unassisted"], color="blue", alpha=0.3, jitter=True)
sns.stripplot(x=["LLM-Assisted"]*len(df_merged), y=df_merged["MAE_assisted"], color="red", alpha=0.3, jitter=True)

# Compute Stats
m_un, ci_un = mean_ci(df_merged["MAE_unassisted"])
m_as, ci_as = mean_ci(df_merged["MAE_assisted"])

# Error Bars (Mean ± 95% CI)
plt.errorbar("Unassisted", m_un, yerr=ci_un, fmt='o', color="blue", capsize=5, elinewidth=3, label="Unassisted Mean")
plt.errorbar("LLM-Assisted", m_as, yerr=ci_as, fmt='o', color="red", capsize=5, elinewidth=3, label="Assisted Mean")

# Boxplot for distribution density
sns.boxplot(data=df_merged[["MAE_unassisted", "MAE_assisted"]], orient="v", showfliers=False, boxprops=dict(alpha=0.2))

# Paired t-test
t_stat, p_val = stats.ttest_rel(df_merged["MAE_unassisted"], df_merged["MAE_assisted"])
plt.text(0.5, m_un, f"Paired t-test p = {p_val:.4f}", fontsize=FONT_SIZE, color="red", ha="center", weight="bold")

# Formatting
plt.ylabel("Mean Absolute Error (MAE)", fontsize=16)
plt.title(f"Comparison of Prediction Error (MAE)\nN = {len(df_merged)}", fontsize=18, fontweight="bold")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Save
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "MAE_Comparison_HighRes.png"), dpi=600)
print(f"✅ MAE paired comparison saved to {OUTPUT_DIR}")
