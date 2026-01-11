import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# ✅ Load the merged results from Step 18
INPUT_FILE = "merged_results_metadata.csv"

if not os.path.exists(INPUT_FILE):
    print(f"❌ ERROR: {INPUT_FILE} not found. Ensure Step 18 ran successfully!")
else:
    df = pd.read_csv(INPUT_FILE)

    # Scatter plot data
    x = df["NumCategoricalVars"]
    y = df["SubjectVariableRatio"]

    # Define inverse function for curve fit
    def inverse_func(x, a, b):
        return a / (x + b)

    # Fit the curve
    params, _ = curve_fit(inverse_func, x, y, maxfev=10000)

    # Generate fitted values for plotting
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = inverse_func(x_fit, *params)

    # **Create the plot**
    plt.figure(figsize=(12, 6), dpi=600)

    # **Plot actual data points**
    plt.scatter(x, y, color="cornflowerblue", label="Synthetic Datasets", alpha=0.8, s=60)

    # **Plot fitted trend line**
    plt.plot(x_fit, y_fit, color="red", linewidth=2.5, label="Complexity Trend")

    # **Special Point (UCI Davis)**
    plt.scatter(15, 23, color="green", s=120, zorder=3, edgecolors="black")
    plt.text(15, 21, "UCI Davis Data", fontsize=12, verticalalignment="top", ha="center", fontweight="bold")

    # **Labels and Formatting**
    plt.xlabel("Number of Categorical Variables", fontsize=16)
    plt.ylabel("Subject-to-Variable Ratio", fontsize=16)
    plt.title("Subject-to-Variable Ratio vs. Number of Categorical Variables", fontsize=16, fontweight="bold")

    # **Custom Legend Handling**
    handles, labels = plt.gca().get_legend_handles_labels()
    # Add manual entry for the Infective Endocarditis comparison
    ie_marker = plt.Line2D([0], [0], color="black", lw=0, marker="o", markersize=8)
    handles.append(ie_marker)
    labels.append("Infective Endocarditis SVR (raw 7.6, after encoding 1.5)")
    
    plt.legend(handles, labels, fontsize=12, loc="upper right", title="Legend", title_fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)

    # **Save the figure as a high-resolution JPEG**
    save_path = "Informatics fig 3 600 dpi.jpg"
    plt.tight_layout()
    plt.savefig(save_path, format="jpeg", dpi=600, bbox_inches="tight")

    print(f"✅ Final figure saved as '{save_path}' (600 DPI).")
