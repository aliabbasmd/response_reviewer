import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

# File paths - standardized for our pipeline
unassisted_file = "numeric_x3_results_extended/unassisted_model_results_extended.csv"
cluster_similarity_file = "cluster_similarity_results_augmented.csv"
output_file = "merged_results_metadata.csv"

if not os.path.exists(unassisted_file) or not os.path.exists(cluster_similarity_file):
    print("❌ Error: Missing input files. Ensure Step 16 and 17 ran successfully.")
else:
    # Load datasets
    df_unassisted = pd.read_csv(unassisted_file)
    df_clusters = pd.read_csv(cluster_similarity_file)

    # Merge datasets on the "Dataset" column
    df_merged = pd.merge(df_unassisted, df_clusters, on="Dataset", how="inner")

    # Filter for R² > 0.2 to focus on meaningful models
    df_filtered = df_merged[df_merged["R2"] > 0.2].copy()

    # Calculate Pearson Correlation
    r_subject, p_subject = stats.pearsonr(df_filtered["SubjectVariableRatio"], df_filtered["R2"])
    r_categorical, p_categorical = stats.pearsonr(df_filtered["NumCategoricalVars"], df_filtered["R2"])

    n_obs = len(df_filtered)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Structural Sensitivity: Predicting R² Success (N = {n_obs})", 
                 fontsize=18, fontweight="bold")

    # Plot 1: R² vs Subject Variable Ratio
    sns.regplot(x="SubjectVariableRatio", y="R2", data=df_filtered, 
                ax=axes[0], scatter_kws={'s': 50, 'alpha':0.6}, line_kws={'color': 'red'})
    axes[0].axvspan(0, 20, color='red', alpha=0.1, label="Low Sample Density")
    axes[0].set_title(f"R² vs SVR\nr = {r_subject:.2f}, p = {p_subject:.4f}", fontsize=15)
    axes[0].set_xlabel("Subject to Variable Ratio (SVR)", fontsize=14)
    axes[0].set_ylabel("R² Score", fontsize=14)

    # Plot 2: R² vs Number of Categorical Variables
    sns.regplot(x="NumCategoricalVars", y="R2", data=df_filtered, 
                ax=axes[1], scatter_kws={'s': 50, 'alpha':0.6}, line_kws={'color': 'blue'})
    axes[1].axvspan(20, df_filtered["NumCategoricalVars"].max(), color='blue', alpha=0.1, label="High Categorical Complexity")
    axes[1].set_title(f"R² vs Categorical Count\nr = {r_categorical:.2f}, p = {p_categorical:.4f}", fontsize=15)
    axes[1].set_xlabel("Number of Categorical Variables", fontsize=14)
    axes[1].set_ylabel("R² Score", fontsize=14)

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("structural_sensitivity_analysis.png")
    print("📊 Sensitivity analysis saved to structural_sensitivity_analysis.png")
    
    # Save merged data for future steps
    df_merged.to_csv(output_file, index=False)
    print(f"✅ Merged metadata saved to {output_file}")
