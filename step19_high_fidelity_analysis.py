import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# **File Paths - Aligned with Master Pipeline**
UNASSISTED_FILE = "numeric_x3_results_extended/unassisted_model_results_extended.csv"
CLUSTER_FILE = "cluster_similarity_results_augmented.csv"
OUTPUT_PLOTS_DIR = "comparison_plots"

# Ensure output directory exists
if not os.path.exists(OUTPUT_PLOTS_DIR):
    os.makedirs(OUTPUT_PLOTS_DIR)

# **Load Data**
if not os.path.exists(UNASSISTED_FILE) or not os.path.exists(CLUSTER_FILE):
    print("❌ Error: Missing input files. Ensure Step 16, 17, and 18 ran successfully.")
else:
    df_unassisted = pd.read_csv(UNASSISTED_FILE)
    df_clusters = pd.read_csv(CLUSTER_FILE)

    # **Merge datasets on the "Dataset" column**
    df_merged = pd.merge(df_unassisted, df_clusters, on="Dataset", how="inner")

    # **Filter for R² > 0.2**
    df_merged = df_merged[df_merged["R2"] > 0.2]

    # **Calculate Pearson Correlation**
    r_subject, p_subject = stats.pearsonr(df_merged["SubjectVariableRatio"], df_merged["R2"])
    r_categorical, p_categorical = stats.pearsonr(df_merged["NumCategoricalVars"], df_merged["R2"])

    n_obs = len(df_merged)

    # **Create figure with high resolution**
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=600)

    fig.suptitle(f"Results of LLM Unassisted Machine Learning (Filtered for R² > 0.2)\n"
                 f"From 100 Synthetic Dataframes (N = {n_obs})", 
                 fontsize=18, fontweight="bold")

    # **Plot 1: R² vs Subject Variable Ratio**
    sns.regplot(x="SubjectVariableRatio", y="R2", data=df_merged, 
                ax=axes[0], scatter_kws={'s': 50, 'alpha':0.6}, line_kws={'color': 'red'})
    axes[0].axvspan(0, 20, color='red', alpha=0.1, label="Less Reliable R²")
    axes[0].set_title(f"R² vs SVR\nr = {r_subject:.2f}, p = {p_subject:.4f}", fontsize=16, fontweight="bold")
    axes[0].set_xlabel("Subject Variable Ratio", fontsize=16)
    axes[0].set_ylabel("R² Score", fontsize=16)

    # **Plot 2: R² vs Number of Categorical Variables**
    sns.regplot(x="NumCategoricalVars", y="R2", data=df_merged, 
                ax=axes[1], scatter_kws={'s': 50, 'alpha':0.6}, line_kws={'color': 'blue'})
    axes[1].axvspan(20, df_merged["NumCategoricalVars"].max(), color='blue', alpha=0.1, label="Less Reliable R²")
    axes[1].set_title(f"R² vs Categorical Vars\nr = {r_categorical:.2f}, p = {p_categorical:.4f}", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Number of Categorical Variables", fontsize=16)
    axes[1].set_ylabel("R² Score", fontsize=16)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # **Save Plot in both JPEG and PNG formats at 600 DPI**
    jpeg_path = os.path.join(OUTPUT_PLOTS_DIR, "r2_analysis_high_res.jpg")
    png_path = os.path.join(OUTPUT_PLOTS_DIR, "r2_analysis_high_res.png")

    plt.savefig(jpeg_path, format="jpeg", dpi=600, bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=600, bbox_inches="tight")

    print(f"\n✅ High-fidelity plots saved in /{OUTPUT_PLOTS_DIR}")
