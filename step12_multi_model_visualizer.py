import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

# --- SETTINGS ---
input_dir = "generated_data"
output_dir = "generated_data"

def compute_similarity(original_data, embedding_data):
    sample_size = min(300, original_data.shape[0], embedding_data.shape[0])
    orig_s = original_data[:sample_size]
    emb_s = embedding_data[:sample_size]
    
    dist_orig = squareform(pdist(orig_s, metric='euclidean'))
    dist_emb = squareform(pdist(emb_s, metric='euclidean'))

    min_sz = min(dist_orig.shape[0], dist_emb.shape[0])
    dist_orig, dist_emb = dist_orig[:min_sz, :min_sz], dist_emb[:min_sz, :min_sz]
    
    corr, _ = spearmanr(dist_orig.flatten(), dist_emb.flatten())
    cos_sim = cosine_similarity(dist_orig.flatten().reshape(1, -1), dist_emb.flatten().reshape(1, -1))[0][0]
    return round(corr, 3), round(cos_sim, 3)

if __name__ == "__main__":
    # Load original datasets
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    dataframes = {}
    for file in csv_files:
        dataset_name = os.path.splitext(file)[0]
        df = pd.read_csv(os.path.join(input_dir, file))
        dataframes[dataset_name] = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)

    # Load embeddings (.npy files)
    embedding_files = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
    embeddings_data = {}
    for file in embedding_files:
        parts = file.replace("embeddings_", "").replace(".npy", "").split("_")
        model_name, dataset_name = parts[0], "_".join(parts[1:])
        
        if dataset_name in dataframes:
            embs = np.load(os.path.join(output_dir, file)).astype(np.float32)
            if embs.ndim == 1: embs = embs.reshape(-1, 1)
            
            # PCA Alignment
            target_dim = dataframes[dataset_name].shape[1]
            if embs.shape[1] > target_dim:
                embs = PCA(n_components=target_dim).fit_transform(embs)
            
            if dataset_name not in embeddings_data: embeddings_data[dataset_name] = {}
            embeddings_data[dataset_name][model_name] = embs

    # Calculate Similarity Scores
    results = []
    for d_name, d_orig in dataframes.items():
        if d_name in embeddings_data:
            for m_name, d_emb in embeddings_data[d_name].items():
                corr, cos = compute_similarity(d_orig, d_emb)
                results.append({"Dataset": d_name, "Model": m_name, "Corr": corr, "Cos": cos})

    sim_df = pd.DataFrame(results).dropna()

    # Visualization Loop
    for m_name in sim_df["Model"].unique():
        m_results = sim_df[sim_df["Model"] == m_name]
        best_case = m_results.loc[m_results["Cos"].idxmax()]
        
        for case_type, case_data in [("BEST", best_case)]:
            d_name = case_data["Dataset"]
            orig_s = dataframes[d_name][:300]
            emb_s = embeddings_data[d_name][m_name][:300]

            # Dimensionality Reduction
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            t_orig = tsne.fit_transform(orig_s)
            t_emb = tsne.fit_transform(emb_s)

            # Clustering
            km = KMeans(n_clusters=4, random_state=42, n_init=10)
            l_orig = km.fit_predict(t_orig)
            l_emb = km.fit_predict(t_emb)

            # Plotting
            fig = plt.figure(figsize=(18, 6))
            
            # 3D View
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.scatter(orig_s[:, 0], orig_s[:, 1], orig_s[:, 2], c=l_orig, cmap='viridis')
            ax1.set_title(f"3D Structure: {d_name}")

            # t-SNE Original
            ax2 = fig.add_subplot(132)
            sns.scatterplot(x=t_orig[:,0], y=t_orig[:,1], hue=l_orig, palette="viridis", ax=ax2)
            ax2.set_title("t-SNE Ground Truth")

            # t-SNE Model
            ax3 = fig.add_subplot(133)
            sns.scatterplot(x=t_emb[:,0], y=t_emb[:,1], hue=l_emb, palette="magma", ax=ax3)
            ax3.set_title(f"{m_name} Perception (CosSim: {case_data['Cos']})")

            plt.tight_layout()
            plt.savefig(f"audit_{m_name}_{case_type}.png")
            print(f"✅ Audit Plot saved for {m_name}")
            plt.close()
