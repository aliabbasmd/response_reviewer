import os
import numpy as np
import pandas as pd
import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Configuration matches previous steps
output_dir = "generated_data"
llms = ["t5", "bert", "roberta"] # Adjust based on what you actually ran in Step 2

def load_embeddings():
    """Loads all saved .npy embeddings and matches them with their metadata."""
    embeddings_data = []
    npy_files = glob.glob(f"{output_dir}/embeddings_*.npy")
    
    for filepath in npy_files:
        # Filename format: embeddings_{llm}_data_{seed}_{eq}.npy
        parts = os.path.basename(filepath).replace(".npy", "").split("_")
        llm_name = parts[1]
        eq_type = parts[-1]
        
        embeddings = np.load(filepath)
        if embeddings.shape[0] > 1:
            embeddings_data.append((llm_name, embeddings, eq_type, filepath))
    return embeddings_data

def reduce_embeddings(embeddings):
    """Reduces high-dim embeddings to 2D using PCA + t-SNE."""
    num_samples, num_features = embeddings.shape
    if num_samples < 5: return None 

    # PCA first to reduce noise
    n_pca = min(50, num_samples, num_features)
    pca_data = PCA(n_components=n_pca).fit_transform(embeddings)

    # t-SNE for 2D visualization/clustering
    perplexity = min(30, num_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(pca_data)
    return tsne

def get_optimal_clusters(tsne_data):
    """Uses the Elbow Method (Inertia) to guess the number of clusters."""
    distortions = []
    max_k = min(10, len(tsne_data))
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(tsne_data)
        distortions.append(kmeans.inertia_)
    
    if len(distortions) < 3: return 1
    # Find the 'elbow' (inflection point)
    kn = np.diff(distortions, 2).argmin() + 2
    return kn

if __name__ == "__main__":
    print("Loading embeddings...")
    data_list = load_embeddings()
    
    results = []
    original_k = 4 # You created 4 clusters: (A-X, A-Y, B-X, B-Y)

    for llm, embs, eq, fname in data_list:
        print(f"Analyzing {llm} embeddings for {eq} data...")
        tsne_res = reduce_embeddings(embs)
        
        if tsne_res is not None:
            computed_k = get_optimal_clusters(tsne_res)
            results.append({
                "LLM": llm,
                "Equation": eq,
                "Computed_Clusters": computed_k,
                "Original_Clusters": original_k
            })

    # Summary Statistics
    df_clusters = pd.DataFrame(results)
    if not df_clusters.empty:
        df_summary = df_clusters.groupby(["LLM", "Equation"]).agg(["mean", "std"]).reset_index()
        print("\n--- Cluster Analysis Summary ---")
        print(df_summary)
        df_summary.to_csv("cluster_summary.csv", index=False)
        print(f"\n✅ Summary saved to cluster_summary.csv")
    else:
        print("No embeddings found. Did you run Step 2?")
