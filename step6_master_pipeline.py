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

def get_yes_no_input(prompt):
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ["yes", "y"]: return True
        if user_input in ["no", "n"]: return False
        print("⚠️ Invalid input! Please enter 'yes' or 'no'.")

RE_RUN = get_yes_no_input("🔄 Do you want to re-run clustering analysis? (yes/no): ")

# Define dataset-embedding pairs
processing_sets = [
    {
        "input_dir": "generated_data",
        "output_dir": "generated_data",
        "models": ["bert", "roberta", "t5"]
    },
    {
        "input_dir": "generated_data",
        "output_dir": "generated_data",
        "models": ["llama"]
    }
]

RESULTS_FILE = "cluster_similarity_results_master.csv"
processed_files = set()
if os.path.exists(RESULTS_FILE) and not RE_RUN:
    existing_results = pd.read_csv(RESULTS_FILE)
    processed_files = set(zip(existing_results["Dataset"], existing_results["Model"]))

def load_data(input_dir, output_dir, models):
    if not os.path.exists(input_dir) or not os.path.exists(output_dir):
        print(f"❌ Missing directory: {input_dir}")
        return {}, {}

    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    dataframes = {os.path.splitext(f)[0]: pd.read_csv(os.path.join(input_dir, f)) for f in csv_files}
    
    embeddings_data = {}
    existing_outputs = set(os.listdir(output_dir))

    for dataset_name in dataframes.keys():
        for model_name in models:
            if (dataset_name, model_name) in processed_files: continue

            # Logic to find .h5 for llama or .npy for others
            ext = ".h5" if model_name == "llama" else ".npy"
            pattern = f"embeddings_{model_name}_{dataset_name}{ext}"
            
            # Simplified matching for your specific naming convention
            matches = [f for f in existing_outputs if dataset_name in f and model_name in f and f.endswith(ext)]
            
            if matches:
                path = os.path.join(output_dir, matches[0])
                try:
                    if ext == ".h5":
                        with h5py.File(path, 'r') as h5f: embs = h5f['embeddings'][:]
                    else:
                        embs = np.load(path)
                    
                    if embs.ndim > 2: embs = embs.reshape(embs.shape[0], -1)
                    if dataset_name not in embeddings_data: embeddings_data[dataset_name] = {}
                    embeddings_data[dataset_name][model_name] = embs
                except Exception as e:
                    print(f"❌ Error loading {path}: {e}")
    return dataframes, embeddings_data

def compute_cluster_similarity(dataframes, embeddings_data):
    results = []
    imputer = SimpleImputer(strategy="mean")
    
    for dataset_name, df in dataframes.items():
        if dataset_name not in embeddings_data: continue
        
        feats = imputer.fit_transform(df.iloc[:, :3].values[:300])
        n_clusters = max(2, min(10, len(np.unique(feats, axis=0))))
        
        kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats)
        dist_orig = squareform(pdist(kmeans_orig.cluster_centers_)).flatten()

        for model, embs in embeddings_data[dataset_name].items():
            print(f"📊 Analyzing {model} on {dataset_name}...")
            embs_sample = imputer.fit_transform(embs[:300])
            pca_embs = PCA(n_components=min(50, embs_sample.shape[1])).fit_transform(embs_sample)
            tsne_embs = TSNE(n_components=2, perplexity=min(30, len(pca_embs)-1), random_state=42).fit_transform(pca_embs)
            
            kmeans_emb = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(tsne_embs)
            dist_emb = squareform(pdist(kmeans_emb.cluster_centers_)).flatten()

            corr, _ = spearmanr(dist_orig, dist_emb)
            results.append({
                "Dataset": dataset_name,
                "Model": model,
                "Distance Correlation": round(corr, 3)
            })
    return results

if __name__ == "__main__":
    all_results = []
    for set_info in processing_sets:
        df_dict, emb_dict = load_data(set_info["input_dir"], set_info["output_dir"], set_info["models"])
        all_results.extend(compute_cluster_similarity(df_dict, emb_dict))

    final_df = pd.DataFrame(all_results)
    if os.path.exists(RESULTS_FILE) and not RE_RUN:
        final_df = pd.concat([pd.read_csv(RESULTS_FILE), final_df]).drop_duplicates()
    
    final_df.to_csv(RESULTS_FILE, index=False)
    print(f"✅ Pipeline complete. Results saved to {RESULTS_FILE}")
