import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# ✅ Load the results file
RESULTS_FILE = "cluster_similarity_results.csv"

if not os.path.exists(RESULTS_FILE):
    print(f"❌ ERROR: {RESULTS_FILE} not found. Please run Step 7 first!")
else:
    results_df = pd.read_csv(RESULTS_FILE)

    # ✅ Ensure that the dataset column is properly formatted
    if "Dataset" in results_df.columns:
        results_df["Function Type"] = results_df["Dataset"].str.extract(r'_(linear|exponential|quadratic|cubic)')
    else:
        results_df["Function Type"] = "unknown"

    def compute_summary(df, metric):
        """Compute mean, std, and 95% confidence interval."""
        summary_df = df.groupby(["Model", "Function Type"])[metric].agg(["mean", "std", "count"]).reset_index()
        summary_df["sem"] = summary_df["std"] / np.sqrt(summary_df["count"])
        # 95% CI using t-distribution
        summary_df["ci95"] = summary_df["sem"] * stats.t.ppf(0.975, df=summary_df["count"]-1)
        return summary_df

    distance_summary = compute_summary(results_df, "Distance Correlation")
    cosine_summary = compute_summary(results_df, "Cosine Similarity")

    def plot_error_bars(summary_df, metric_name, y_label, y_lim=None):
        function_types = summary_df["Function Type"].dropna().unique()
        
        for function in function_types:
            plt.figure(figsize=(12, 7))
            subset = summary_df[summary_df["Function Type"] == function].sort_values(by="mean", ascending=False)

            # ✅ Plot Mean with 95% CI
            plt.errorbar(subset["Model"], subset["mean"], yerr=subset["ci95"], fmt='o', color='red', ecolor='black', capsize=5, label="Mean ± 95% CI")

            plt.xlabel("Model", fontsize=16)
            plt.ylabel(y_label, fontsize=16)
            plt.title(f"{y_label} (95% CI) - {function.capitalize()} Functions", fontsize=18)
            
            if y_lim:
                plt.ylim(y_lim)

            plt.xticks(rotation=45, fontsize=14)
            plt.yticks(fontsize=14)

            # Add data labels
            for i, (idx, row) in enumerate(subset.iterrows()):
                plt.text(i, row["mean"] + row["ci95"] + 0.01, f"{row['mean']:.3f}", ha='center', fontweight='bold')

            plt.legend(fontsize=14)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            
            # Save file
            save_path = f"ci95_{function}_{metric_name.replace(' ', '_')}.png"
            plt.savefig(save_path)
            print(f"📊 Saved Confidence Interval plot: {save_path}")
            plt.close()

    # ✅ Generate the plots
    plot_error_bars(distance_summary, "Distance Correlation", "Distance Correlation Mean")
    plot_error_bars(cosine_summary, "Cosine Similarity", "Cosine Similarity Mean", y_lim=(0.6, 0.85))
