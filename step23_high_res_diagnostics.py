import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# **File Paths**
DATA_FILE = "merged_results_metadata.csv"
OUTPUT_PLOTS_DIR = "comparison_plots"

# Ensure output directory exists
if not os.path.exists(OUTPUT_PLOTS_DIR):
    os.makedirs(OUTPUT_PLOTS_DIR)

# **Load Data**
if not os.path.exists(DATA_FILE):
    print(f"❌ ERROR: {DATA_FILE} not found. Run Step 18 first!")
else:
    df_merged = pd.read_csv(DATA_FILE)

    # **Filter for R² > 0.2**
    df_merged = df_merged[df_merged["R2"] > 0.2].copy()

    # **Regression Model Predicting R²**
    X = df_merged[["SubjectVariableRatio", "NumCategoricalVars"]]
    X = sm.add_constant(X)
    y = df_merged["R2"]
    model = sm.OLS(y, X).fit()

    # **Compute Residuals**
    df_merged["Residuals"] = model.resid

    # **Breusch-Pagan Test**
    bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, X)
    print(f"\n📈 Breusch-Pagan p-value: {bp_pval:.4f}")

    # **Plotting with High Resolution**
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=600)

    # Plot 1: SVR Residuals
    sns.scatterplot(x="SubjectVariableRatio", y="Residuals", data=df_merged, ax=axes[0], color="blue", alpha=0.6)
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    axes[0].axvline(x=20, color="black", linestyle="--", linewidth=2, label="Threshold: SVR=20")
    axes[0].set_title("Residuals vs SVR", fontsize=16, fontweight="bold")
    axes[0].set_xlabel("Subject Variable Ratio", fontsize=14)
    axes[0].legend(fontsize=12)

    # Plot 2: Categorical Residuals
    sns.scatterplot(x="NumCategoricalVars", y="Residuals", data=df_merged, ax=axes[1], color="green", alpha=0.6)
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    axes[1].axvline(x=20, color="black", linestyle="--", linewidth=2, label="Threshold: Cats=20")
    axes[1].set_title("Residuals vs Categorical Count", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Number of Categorical Variables", fontsize=14)
    axes[1].legend(fontsize=12)

    plt.tight_layout()

    # **Save high-res assets**
    jpeg_path = os.path.join(OUTPUT_PLOTS_DIR, "residuals_analysis_600dpi.jpg")
    png_path = os.path.join(OUTPUT_PLOTS_DIR, "residuals_analysis_600dpi.png")

    plt.savefig(jpeg_path, format="jpeg", dpi=600, bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=600, bbox_inches="tight")

    print(f"✅ Diagnostic reports saved to /{OUTPUT_PLOTS_DIR}")
