import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Set consistent path
output_dir = "generated_data"

def reduce_embeddings(embeddings):
    num_samples, num_features = embeddings.shape
    if num_samples < 2: return None

    n_components = min(50, num_samples, num_features)
    if n_components < 2: return None

    pca = PCA(n_components=n_components).fit_transform(embeddings)
    perplexity = min(30, num_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(pca)
    return tsne

def optimal_clusters(tsne_data):
    distortions = []
    max_k = min(11, len(tsne_data))
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(tsne_data)
        distortions.append(kmeans.inertia_)
    if len(distortions) < 3: return 1
    return np.diff(distortions, 2).argmin() + 2

def plot_tsne(tsne_data, clusters, llm_name, filename):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=clusters, palette="viridis", s=60, alpha=0.7)
    plt.title(f"t-SNE Clusters: {llm_name}\nSource: {filename}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    # Save the file so it's viewable in WSL/Windows
    save_path = f"plot_{llm_name}.png"
    plt.savefig(save_path)
    print(f"📊 Plot saved as {save_path}")
    plt.close()

if __name__ == "__main__":
    # Find all embedding files
    embedding_files = glob.glob(f"{output_dir}/embeddings_*.npy")
    
    if not embedding_files:
        print("❌ No embeddings found. Please run Step 2 first.")
    else:
        # Process the found files
        for filepath in embedding_files[:3]: # Limit to first 3 for speed
            file_base = os.path.basename(filepath)
            llm_name = file_base.split("_")[1]
            
            print(f"Processing visualization for {llm_name}...")
            embeddings = np.load(filepath)
            tsne_data = reduce_embeddings(embeddings)

            if tsne_data is not None:
                k = optimal_clusters(tsne_data)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(tsne_data)
                plot_tsne(tsne_data, kmeans.labels_, llm_name, file_base)
