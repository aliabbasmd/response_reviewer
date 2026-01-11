import numpy as np
import pandas as pd
import os
import random
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a directory for generated data
output_dir = "generated_data"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

def generate_data(seed):
    """Generates synthetic data with continuous and categorical variables."""
    np.random.seed(seed)
    random.seed(seed)

    x1 = np.random.uniform(-10, 10, 500)
    x2 = np.random.uniform(-10, 10, 500)
    
    equation_type = random.choice(["linear", "quadratic", "cubic", "exponential"])
    
    if equation_type == "linear":
        x3 = 3*x1 + 2*x2 + np.random.normal(0, 5, 500)
    elif equation_type == "quadratic":
        x3 = x1**2 - 2*x2 + np.random.normal(0, 5, 500)
    elif equation_type == "cubic":
        x3 = x1**3 - x2**2 + np.random.normal(0, 5, 500)
    else:
        x3 = np.exp(0.1*x1) - np.exp(0.1*x2) + np.random.normal(0, 5, 500)

    cat1 = np.random.choice(["A", "B"], 500)
    cat2 = np.random.choice(["X", "Y"], 500)
    
    x1 += (cat1 == "B") * 50
    x2 += (cat2 == "Y") * 50

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat1": cat1, "cat2": cat2})
    
    filename = f"{output_dir}/data_{seed}_{equation_type}.csv"
    df.to_csv(filename, index=False)
    return df, equation_type, filename

def plot_3d_scatter(df, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    color_map = {'A': 'blue', 'B': 'red'}
    marker_map = {'X': 'o', 'Y': 's'}
    
    for (cat1_value, cat2_value), subset in df.groupby(["cat1", "cat2"]):
        ax.scatter(
            subset['x1'], subset['x2'], subset['x3'], 
            label=f"{cat1_value}-{cat2_value}",
            color=color_map[cat1_value],
            marker=marker_map[cat2_value],
            s=60, alpha=0.8, edgecolors="black"
        )
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title(title)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    print("Generating 100 datasets in /generated_data...")
    datasets_info = [generate_data(seed) for seed in range(100)]
    sample_df, eq_type, sample_filename = datasets_info[0]
    print(f"Dataset generation complete. Example saved as: {sample_filename}")
    # plot_3d_scatter(sample_df, f"3D Scatter Plot - {eq_type}") # Uncomment if using a GUI-enabled WSL
