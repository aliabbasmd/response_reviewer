import os
import gc
import sys
import h5py
import numpy as np
import pandas as pd
import random
import shap
from glob import glob
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
GENERATED_DATA_DIR = "generated_data"
EMBEDDINGS_DIRS = {
    "bert": "generated_data",
    "roberta": "generated_data",
    "gatortron": "generated_data",
    "t5": "generated_data",
    "ernie": "generated_data",
    "minilm": "generated_data",
    "e5_small": "generated_data",
    "llama": "generated_data"
}
CLUSTER_RESULTS_PATH = "cluster_similarity_results.csv"
OUTPUT_DATA_DIR = "generated_data_with_clusters"
PROGRESS_FILE = "cluster_assignment_progress.csv"
UNASSISTED_FILE = "unassisted_model_results.csv"
RANDOM_SEED = 42

# --- HELPER FUNCTIONS ---

def load_datasets_info(data_dir):
    info_dict = {}
    csv_files = glob(os.path.join(data_dir, "data_*.csv"))
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        ds_name = os.path.splitext(filename)[0]
        df = pd.read_csv(csv_path)
        cat_count = sum((df[col].dtype == "object") for col in df.columns)
        info_dict[ds_name] = (df.shape[0], df.shape[1], cat_count)
    return info_dict

def load_embeddings(ds_name, model_name):
    base_dir = EMBEDDINGS_DIRS.get(model_name, "generated_data")
    if model_name == "llama":
        files = glob(os.path.join(base_dir, f"*{ds_name}*.h5"))
        if not files: return None
        with h5py.File(files[0], 'r') as h5f: return h5f['embeddings'][:]
    else:
        files = glob(os.path.join(base_dir, f"embeddings_{model_name}_{ds_name}*.npy"))
        if not files: return None
        return np.load(files[0])

# --- ML ENGINE ---

def evaluate_regression(X, y, numeric_cols, cat_cols):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    
    models = {
        "LinearReg": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    }
    
    best_name, best_score, best_model = None, -np.inf, None
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        score = r2_score(y_test, pipe.predict(X_test))
        if score > best_score:
            best_score, best_name, best_model = score, name, pipe
    return best_name, best_score, best_model

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DATA_DIR): os.makedirs(OUTPUT_DATA_DIR)
    
    print("🚀 Starting Augmentation and ML Pipeline...")
    data_info = load_datasets_info(GENERATED_DATA_DIR)
    
    # Process Clustering & Regression
    all_ml_results = []
    for ds_name, path in [(n, os.path.join(GENERATED_DATA_DIR, n+".csv")) for n in data_info.keys()]:
        print(f"Processing: {ds_name}")
        df = pd.read_csv(path)
        
        # Regression Logic (Unassisted)
        numeric_cols = ["x1", "x2"]
        cat_cols = [c for c in df.columns if c.startswith("cat")]
        X, y = df[numeric_cols + cat_cols], df["x3"]
        
        m_name, m_score, m_pipe = evaluate_regression(X, y, numeric_cols, cat_cols)
        all_ml_results.append({"Dataset": ds_name, "BestModel": m_name, "R2_Score": m_score})
        
        # Save updated CSV
        df.to_csv(os.path.join(OUTPUT_DATA_DIR, f"{ds_name}.csv"), index=False)

    pd.DataFrame(all_ml_results).to_csv(UNASSISTED_FILE, index=False)
    print(f"✅ Finished. Results saved to {UNASSISTED_FILE}")
