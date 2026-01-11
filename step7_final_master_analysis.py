import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

# --- SETTINGS ---
RE_RUN = True  
RESULTS_FILE = "cluster_similarity_results.csv"

# Update these paths if they differ on your WSL machine
processing_sets = [
    {
        "input_dir": "/home/abbasali/synthetic data/generated_data",
        "output_dir": "/home/abbasali/synthetic data/embeddings",
        "file_ext": ".npy",
        "models": ["bert", "ernie", "gatortron", "roberta", "t5", "e5_small", "minilm"]
    },
    {
        "input_dir": "/home/abbasali/synthetic data/generated_data_llama",
        "output_dir": "/home/abbasali/synthetic data/embeddings_llama",
        "file_ext": ".h5",
        "models": ["llama"]
    }
]

def load_data(input_dir, output_dir, file_ext, models):
    print(f"\n📂 Scanning: {input_dir}")
    if not os.path.exists(input_dir) or not os.path.exists(output_dir):
        print(f"❌ Directory error. Skipping this set.")
        return {}, {}

    existing_files = set(os.listdir(output_dir))
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    dataframes = {os.path.splitext(f)[0]: pd.read_csv(os.path.join(input_dir, f)) for f in csv_files}
    
    embeddings_data = {}
    for filename in csv_files:
        dataset_name = os.path.splitext(filename)[0]
        for model_name in models:
            if model_name == "llama":
                matches = [f for f in existing_files if dataset_name in f and f.endswith(".h5")]
            else:
                matches = [f for f in existing_files if f.startswith(f"embeddings_{model_name}_{dataset_name}") and f.endswith(".npy")]

            if not matches: continue
            
            path = os.path.join(output_dir, matches[0])
            try:
                if path.endswith(".h5"):
                    with h5py.File(path, 'r') as h5f: embs = h5f['embeddings'][:]
                else:
                    embs = np.load(path)
                
                if dataset_name not in embeddings_data: embeddings_data[dataset_name] = {}
                embeddings_data[dataset_name][model_name] = embs
            except Exception as e:
                print(f"❌ Failed to load {path}: {e}")
    return dataframes, embeddings_data

def compute_cluster_similarity(dataframes, embeddings_data):
    results = []
    imputer = SimpleImputer(strategy="mean")
    
    for dataset_name, df in dataframes.items():
        if dataset_name not in embeddings_data or df.empty: continue
        print(f"🔍 Analyzing Structural Similarity: {dataset_name}")

        feats = imputer.fit_transform(df.iloc[:, :3].values[:300])
        n_clusters = max(2, min(10, len(np.unique(feats, axis=0))))
        kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats)
        dist_orig = squareform(pdist(kmeans_orig.cluster_centers_)).flatten()

        for model, embs in embeddings_data[dataset_name].items():
            if embs.ndim > 2: embs = embs.reshape(embs.shape[0], -1)
            
            embs_sample = imputer.fit_transform(embs[:300])
            pca_embs = PCA(n_components=min(50, embs_sample.shape[1])).fit_transform(embs_sample)
            tsne_embs = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(pca_embs)
            
            adj_clusters = max(2, min(n_clusters, len(tsne_embs)))
            kmeans_emb = KMeans(n_clusters=adj_clusters, random_state=42, n_init=10).fit(tsne_embs)
            dist_emb = squareform(pdist(kmeans_emb.cluster_centers_)).flatten()

            min_len = min(len(dist_orig), len(dist_emb))
            corr, _ = spearmanr(dist_orig[:min_len], dist_emb[:min_len])
            results.append({
                "Dataset": dataset_name,
                "Model": model,
                "Distance Correlation": round(corr, 3)
            })
    return results

if __name__ == "__main__":
    all_final_results = []
    for set_info in processing_sets:
        dfs, embs = load_data(set_info["input_dir"], set_info["output_dir"], set_info["file_ext"], set_info["models"])
        if embs:
            all_final_results.extend(compute_cluster_similarity(dfs, embs))

    res_df = pd.DataFrame(all_final_results)
    if not res_df.empty:
        res_df.to_csv(RESULTS_FILE, index=False)
        print(f"\n✅ Step 7 Complete: Final report saved to {RESULTS_FILE}")
