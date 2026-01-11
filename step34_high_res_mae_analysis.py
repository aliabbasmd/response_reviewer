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

# --- LOAD & FILTER ---
df_as = pd.read_csv(ASSISTED_FILE)
df_un = pd.read_csv(UNASSISTED_FILE)

# Filtering for high-fidelity models only (R2 > 0.6)
df_as = df_as[df_as["R2"] > 0.6]
df_un = df_un[df_un["R2"] > 0.6]

df_merged = pd.merge(df_un, df_as, on="Dataset", suffixes=("_unassisted", "_assisted"))

def mean_ci(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
    return mean, margin

# --- PLOTTING ---
plt.figure(figsize=(12, 8), dpi=600)

# Paired Trajectories
for i in range(len(df_merged)):
    plt.plot(["Unassisted", "LLM-Assisted"], 
             [df_merged.iloc[i]["MAE_unassisted"], df_merged.iloc[i]["MAE_assisted"]],
             linestyle="dashed", color="gray", alpha=0.4)

sns.stripplot(x=["Unassisted"]*len(df_merged), y=df_merged["MAE_unassisted"], jitter=True, alpha=0.3, color="blue")
sns.stripplot(x=["LLM-Assisted"]*len(df_merged), y=df_merged["MAE_assisted"], jitter=True, alpha=0.3, color="red")

m_un, ci_un = mean_ci(df_merged["MAE_unassisted"])
m_as, ci_as = mean_ci(df_merged["MAE_assisted"])

plt.errorbar("Unassisted", m_un, yerr=ci_un, fmt='o', color="blue", capsize=5, elinewidth=3, label="Unassisted Mean ± 95% CI")
plt.errorbar("LLM-Assisted", m_as, yerr=ci_as, fmt='o', color="red", capsize=5, elinewidth=3, label="Assisted Mean ± 95% CI")

ax = sns.boxplot(data=df_merged[["MAE_unassisted", "MAE_assisted"]], orient="v", showfliers=False, boxprops=dict(alpha=0.3), medianprops=dict(color="black"))

# Stats
t_stat, p_val = stats.ttest_rel(df_merged["MAE_unassisted"], df_merged["MAE_assisted"])

plt.ylabel("Mean Absolute Error (MAE)", fontsize=FONT_SIZE)
plt.title(f"Comparative MAE Analysis (N = {len(df_merged)})\nHigh-Fidelity Models ($R^2 > 0.6$)", fontsize=FONT_SIZE + 2, fontweight="bold")
plt.gcf().text(0.5, 0.02, f"Paired t-test p = {p_val:.6f}", fontsize=FONT_SIZE, color="red", ha="center", weight="bold")

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "MAE_Analysis_600dpi.png")
plt.savefig(save_path, dpi=600, bbox_inches="tight")
print(f"✅ 600 DPI Plot saved: {save_path}")
