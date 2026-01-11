import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# ✅ Load the merged results
INPUT_FILE = "merged_results_metadata.csv"

if not os.path.exists(INPUT_FILE):
    print(f"❌ ERROR: {INPUT_FILE} not found. Please run Step 18 first!")
else:
    df = pd.read_csv(INPUT_FILE)

    # Scatter plot data
    x = df["NumCategoricalVars"]
    y = df["SubjectVariableRatio"]

    # Define inverse function for curve fit
    def inverse_func(x, a, b):
        return a / (x + b)

    # Fit the curve (Inverse relationship)
    params, _ = curve_fit(inverse_func, x, y, maxfev=10000)

    # Generate fitted values
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = inverse_func(x_fit, *params)

    # Create the plot
    plt.figure(figsize=(12, 6), dpi=300)

    # Plot actual data points
    plt.scatter(x, y, color="cornflowerblue", label="Synthetic Datasets", alpha=0.6, s=60)

    # Plot fitted trend line
    plt.plot(x_fit, y_fit, color="red", linewidth=2.5, label="Complexity Trend (Inverse)")

    # Special Benchmark Points
    # UCI Davis example: 15 categorical vars, ~23 SVR
    plt.scatter(15, 23, color="green", s=120, zorder=5, edgecolors='black')
    plt.text(15, 21, "UCI Davis Benchmark", fontsize=11, fontweight="bold", ha="center", va="top")

    # Labels and Formatting
    plt.xlabel("Number of Categorical Variables", fontsize=14)
    plt.ylabel("Subject-to-Variable Ratio (SVR)", fontsize=14)
    plt.title("The Complexity Frontier: SVR vs. Categorical Sparsity", fontsize=16, fontweight="bold")

    # Custom Legend Handling
    handles, labels = plt.gca().get_legend_handles_labels()
    # Add manual entry for the Infective Endocarditis comparison
    ie_marker = plt.Line2D([0], [0], color="black", lw=0, marker="o", markersize=8)
    handles.append(ie_marker)
    labels.append("Infective Endocarditis (Target: 7.6 SVR, Post-Encoding: 1.5)")

    plt.legend(handles, labels, fontsize=10, loc="upper right", frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    
    # Save the final mapping
    save_path = "dataset_complexity_frontier.png"
    plt.savefig(save_path, dpi=300)
    print(f"📊 Final complexity mapping saved to {save_path}")
