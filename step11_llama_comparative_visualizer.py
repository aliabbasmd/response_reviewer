import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

# --- SETTINGS ---
input_dir = "generated_data"
output_dir = "generated_data" # Change if Llama .h5 files are in a specific subfolder

def compute_similarity(original_data, embedding_data):
    sample_size = min(300, original_data.shape[0], embedding_data.shape[0])
    original_sample = original_data[:sample_size]
    embedding_sample = embedding_data[:sample_size]
    
    dist_orig = squareform(pdist(original_sample, metric='euclidean'))
    dist_emb = squareform(pdist(embedding_sample, metric='euclidean'))

    min_size = min(dist_orig.shape[0], dist_emb.shape[0])
    dist_orig, dist_emb = dist_orig[:min_size, :min_size], dist_emb[:min_size, :min_size]
    
    correlation, _ = spearmanr(dist_orig.flatten(), dist_emb.flatten())
    cosine_sim = cosine_similarity(dist_orig.flatten().reshape(1, -1), dist_emb.flatten().reshape(1, -1))[0][0]
    return round(correlation, 3), round(cosine_sim, 3)

if __name__ == "__main__":
    # Load original datasets
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv") and "llama" in f.lower()]
    if not csv_files:
        print("⚠️ No Llama-specific CSV files found. Checking all CSVs...")
        csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    dataframes = {}
    for file in csv_files:
        dataset_name = os.path.splitext(file)[0]
        df = pd.read_csv(os.path.join(input_dir, file))
        dataframes[dataset_name] = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)

    # Load Llama Embeddings
    embedding_files = [f for f in os.listdir(output_dir) if f.endswith(".h5")]
    embeddings_data = {}

    for file in embedding_files:
        dataset_name = file.replace("embeddings_", "").replace(".h5", "").replace("llama_", "")
        if dataset_name in dataframes:
            with h5py.File(os.path.join(output_dir, file), 'r') as h5f:
                embeddings = np.array(h5f['embeddings'][:], dtype=np.float32)
                
                # PCA reduction to match original feature count
                original_dim = dataframes[dataset_name].shape[1]
                if embeddings.shape[1] > original_dim:
                    pca = PCA(n_components=original_dim)
                    embeddings = pca.fit_transform(embeddings)
                embeddings_data[dataset_name] = embeddings

    # Process and Plot
    similarity_results = []
    for d_name, d_orig in dataframes.items():
        if d_name in embeddings_data:
            d_emb = embeddings_data[d_name]
            corr, cos = compute_similarity(d_orig, d_emb)
            similarity_results.append((d_name, corr, cos))

    # Sort by highest similarity
    sorted_sets = sorted(similarity_results, key=lambda x: x[2], reverse=True)

    for d_name, corr, cos in sorted_sets[:3]: # Limit to top 3 for speed
        print(f"🎨 Generating Side-by-Side Plot for {d_name}...")
        orig = dataframes[d_name][:300]
        emb = embeddings_data[d_name][:300]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_orig = tsne.fit_transform(orig)
        tsne_emb = tsne.fit_transform(emb)

        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels_orig = km.fit_predict(tsne_orig)
        labels_emb = km.fit_predict(tsne_emb)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        sns.scatterplot(ax=axes[0], x=tsne_orig[:, 0], y=tsne_orig[:, 1], hue=labels_orig, palette="viridis", s=80)
        axes[0].set_title(f"Original Structure: {d_name}")
        
        sns.scatterplot(ax=axes[1], x=tsne_emb[:, 0], y=tsne_emb[:, 1], hue=labels_emb, palette="coolwarm", s=80)
        axes[1].set_title(f"Llama Embedding Structure (CosSim: {cos})")

        plt.tight_layout()
        save_path = f"comparison_llama_{d_name}.png"
        plt.savefig(save_path)
        print(f"✅ Saved comparison to {save_path}")
        plt.close()
