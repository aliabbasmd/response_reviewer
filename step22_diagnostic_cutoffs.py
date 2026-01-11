import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import os

# **Load Data**
INPUT_FILE = "merged_results_metadata.csv"

if not os.path.exists(INPUT_FILE):
    print(f"❌ ERROR: {INPUT_FILE} not found. Run Step 18 first!")
else:
    df_merged = pd.read_csv(INPUT_FILE)

    # **Filter for R² > 0.2**
    df_merged = df_merged[df_merged["R2"] > 0.2].copy()

    # **Regression Model Predicting R²**
    X = df_merged[["SubjectVariableRatio", "NumCategoricalVars"]]
    X = sm.add_constant(X)
    y = df_merged["R2"]
    model = sm.OLS(y, X).fit()

    # **Compute Residuals**
    df_merged["Residuals"] = model.resid
    df_merged["Fitted_R2"] = model.fittedvalues

    # **Breusch-Pagan Test**
    bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, X)
    print(f"\n--- Statistical Diagnostics ---")
    print(f"Breusch-Pagan Test Statistic: {bp_stat:.4f}")
    print(f"p-value: {bp_pval:.4f}")
    if bp_pval < 0.05:
        print("Result: Significant Heteroscedasticity detected. Cutoffs are justified.")
    else:
        print("Result: Variance is homoscedastic. Cutoffs may be exploratory.")

    # **Plotting**
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Residuals vs SVR
    sns.scatterplot(x="SubjectVariableRatio", y="Residuals", data=df_merged, ax=axes[0], alpha=0.6)
    axes[0].axhline(y=0, color="red", linestyle="--")
    axes[0].axvline(x=20, color="black", linestyle="--", linewidth=2, label="Threshold: SVR=20")
    axes[0].set_title("Residual Variance vs SVR", fontsize=15, fontweight="bold")
    axes[0].set_xlabel("Subject Variable Ratio", fontsize=14)
    axes[0].legend()

    # Plot 2: Residuals vs NumCategoricalVars
    sns.scatterplot(x="NumCategoricalVars", y="Residuals", data=df_merged, ax=axes[1], alpha=0.6)
    axes[1].axhline(y=0, color="red", linestyle="--")
    axes[1].axvline(x=20, color="black", linestyle="--", linewidth=2, label="Threshold: Cats=20")
    axes[1].set_title("Residual Variance vs Categorical Count", fontsize=15, fontweight="bold")
    axes[1].set_xlabel("Number of Categorical Variables", fontsize=14)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("diagnostic_residuals_analysis.png", dpi=300)
    print("📊 Diagnostic plots saved to diagnostic_residuals_analysis.png")
