import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the directory containing the generated data files
output_dir = "generated_data"

if not os.path.exists(output_dir):
    print(f"❌ Error: {output_dir} directory not found. Run Step 1 first!")
else:
    # Get the list of all CSV files in the directory
    csv_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])

    # Plot the first 3 datasets to verify (avoiding opening 100 windows)
    for csv_file in csv_files[:3]:
        file_path = os.path.join(output_dir, csv_file)
        print(f"📈 Plotting {csv_file}...")
        df = pd.read_csv(file_path)

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Use the categorical variables to color if they exist
        # This helps see the 4 clusters (A-X, A-Y, B-X, B-Y)
        scatter = ax.scatter(df['x1'], df['x2'], df['x3'], c=df['x3'], cmap='viridis', alpha=0.7)
        
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.set_title(f'3D Scatter Plot - {csv_file}')
        
        # Save as PNG for easy viewing in WSL/Windows
        save_name = f"verify_{csv_file.replace('.csv', '.png')}"
        plt.savefig(save_name)
        print(f"✅ Saved plot as {save_name}")
        plt.close()
