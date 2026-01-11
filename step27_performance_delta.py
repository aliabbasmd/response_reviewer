import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# **File Paths**
UNASSISTED_FILE = "numeric_x3_results_extended/unassisted_model_results_extended.csv"
ASSISTED_FILE = "llm_assisted_logistic_top5_model_results.csv"
OUTPUT_DIR = "comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# **Load Results**
if not os.path.exists(UNASSISTED_FILE) or not os.path.exists(ASSISTED_FILE):
    print("❌ Error: Missing result files. Ensure Steps 17 and 26 ran successfully.")
else:
    # We use R2 for unassisted and AUC for assisted - for a fair comparison, 
    # we focus on the relative ranking or standardize if both have AUC.
    # Assuming both now have AUC after your latest script.
    df_un = pd.read_csv(UNASSISTED_FILE)
    df_as = pd.read_csv(ASSISTED_FILE)

    # Standardize columns for merging
    df_as = df_as.rename(columns={"AUC": "AUC_Assisted"})
    # If unassisted has R2, we compare improvements in explained variance or AUC
    # If you updated Step 17 to include AUC, use that.
    df_un = df_un.rename(columns={"R2": "R2_Unassisted"}) 

    # Merge on Dataset
    df_comp = pd.merge(df_as, df_un, on="Dataset", how="inner")

    # Group by Model to see which algorithm benefited most from LLM assistance
    summary = df_comp.groupby("Model")["AUC_Assisted"].mean().reset_index()

    # **Visualization: Assisted Performance Distribution**
    plt.figure(figsize=(12, 6), dpi=300)
    sns.boxplot(x="Model", y="AUC_Assisted", data=df_comp, palette="Set2")
    plt.axhline(y=0.5, color='red', linestyle='--', label="Random Chance")
    plt.title("LLM-Assisted Model Performance (AUC)", fontsize=16, fontweight="bold")
    plt.ylabel("AUC Score", fontsize=14)
    plt.xlabel("Predictive Model", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "assisted_performance_boxplot.png"))
    
    print(f"✅ Comparison report saved to {OUTPUT_DIR}")
