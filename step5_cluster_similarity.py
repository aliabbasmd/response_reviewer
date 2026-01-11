import os
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import h5py

# Adjusted paths for consistency with previous steps
input_dir = "generated_data"
output_dir = "generated_data" # Change this if your .h5 files are elsewhere

# Load all CSV datasets
dataframes = {}
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

for file in csv_files:
    file_path = os.path.join(input_dir, file)
    dataset_name = os.path.splitext(file)[0]
    dataframes[dataset_name] = pd.read_csv(file_path)

# Load all embeddings from HDF5 files
embeddings_data = {}
embedding_files = [f for f in os.listdir(output_dir) if f.endswith(".h5")]

for file in embedding_files:
    # Attempting to match the filename structure
    dataset_name = file.replace("embeddings_llama_", "").replace(".h5", "")
    
    if dataset_name in dataframes:
        if dataset_name not in embeddings_data:
            embeddings_data[dataset_name] = {}
        
        with h5py.File(os.path.join(output_dir, file), 'r') as h5f:
            embeddings_data[dataset_name]['llama'] = h5f['embeddings'][:]

def compute_cluster_similarity(dataset_name):
    results = []
    if dataset_name not in embeddings_data:
        return results

    df = dataframes[dataset_name]
    # Select x1, x2, x3
    continuous_features = df[['x1', 'x2', 'x3']].values
    sample_size = min(300, continuous_features.shape[0])
    continuous_sample = continuous_features[:sample_size]

    fixed_num_clusters = 5 

    # Original Data Clustering
    kmeans_orig = KMeans(n_clusters=fixed_num_clusters, random_state=42, n_init=10).fit(continuous_sample)
    dist_orig = squareform(pdist(kmeans_orig.cluster_centers_, metric='euclidean'))

    # Embedding Clustering
    for model_name, embeddings in embeddings_data[dataset_name].items():
        embeddings_sample = embeddings[:sample_size]
        kmeans_emb = KMeans(n_clusters=fixed_num_clusters, random_state=42, n_init=10).fit(embeddings_sample)
        dist_emb = squareform(pdist(kmeans_emb.cluster_centers_, metric='euclidean'))

        # Correlation between the two distance matrices
        correlation, _ = spearmanr(dist_orig.flatten(), dist_emb.flatten())
        cos_sim = cosine_similarity(dist_orig.flatten().reshape(1, -1), dist_emb.flatten().reshape(1, -1))[0][0]

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Distance Correlation": round(correlation, 3),
            "Cosine Similarity": round(cos_sim, 3)
        })
    return results

if __name__ == "__main__":
    print(f"✅ Found {len(dataframes)} datasets and {len(embedding_files)} embedding files.")
    
    if len(embeddings_data) == 0:
        print("⚠️ No matching embeddings found. Check your .h5 filenames!")
    else:
        num_workers = min(multiprocessing.cpu_count(), len(dataframes))
        with multiprocessing.Pool(num_workers) as pool:
            all_results = pool.map(compute_cluster_similarity, list(dataframes.keys()))

        final_results = [item for sublist in all_results for item in sublist]
        results_df = pd.DataFrame(final_results)

        # Save and Print Summary
        summary = results_df.groupby("Model")[["Distance Correlation", "Cosine Similarity"]].agg(["mean", "std"])
        results_df.to_csv("cluster_similarity_results.csv", index=False)
        print("\n--- Model Similarity Summary ---")
        print(summary)

        # Visualizations
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Model", y="Distance Correlation", data=results_df, palette="viridis")
        plt.title("Structural Alignment: Original Data vs. LLM Embeddings")
        plt.savefig("similarity_boxplot.png")
        print("📊 Visualization saved as similarity_boxplot.png")
