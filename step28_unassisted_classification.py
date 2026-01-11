import os
import numpy as np
import pandas as pd
from glob import glob
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# --- CONFIG ---
DATA_DIR = "generated_data_with_clusters"
OUTPUT_FILE = "cluster_classification_results_unassisted.csv"
RANDOM_SEED = 42

MODELS = {
    "LogisticReg": LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "GradientBoost": GradientBoostingClassifier(random_state=RANDOM_SEED)
}

def build_preprocessor(numeric_cols, cat_cols):
    num_pipe = Pipeline([("num_imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)])

def compute_multi_class_auc(y_true, y_prob):
    try:
        n_classes = len(np.unique(y_true))
        if y_prob.shape[1] != n_classes: return "NA"
        if n_classes == 2:
            return roc_auc_score(y_true, y_prob[:, 1])
        return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except: return "NA"

def train_and_select_best(X, y):
    if len(np.unique(y)) < 2: return None, "NA", None
    
    numeric_cols = [c for c in X.columns if c in ["x1", "x2"]]
    cat_cols = [c for c in X.columns if c not in numeric_cols]
    
    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    X_enc = preprocessor.fit_transform(X, y)
    if hasattr(X_enc, "toarray"): X_enc = X_enc.toarray()
    
    counts = Counter(y)
    min_count = min(counts.values())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.3, random_state=RANDOM_SEED, 
        stratify=(y if min_count >= 2 else None)
    )
    
    # SMOTE to balance clusters
    if min_count > 1:
        k = min(5, min_count - 1)
        X_train, y_train = SMOTE(random_state=RANDOM_SEED, k_neighbors=k).fit_resample(X_train, y_train)

    best_auc, best_name, best_pipe = -1, None, None
    for name, model in MODELS.items():
        try:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)
            auc_v = compute_multi_class_auc(y_test, y_prob)
            if auc_v != "NA" and auc_v > best_auc:
                best_auc, best_name, best_pipe = auc_v, name, model
        except: continue
            
    return best_name, best_auc, best_pipe

if __name__ == "__main__":
    csv_files = glob(os.path.join(DATA_DIR, "*.csv"))
    results = []

    for fpath in csv_files:
        ds_name = os.path.splitext(os.path.basename(fpath))[0]
        print(f"📊 Processing Unassisted Baseline: {ds_name}")
        df = pd.read_csv(fpath)
        
        cluster_cols = [c for c in df.columns if c.startswith("cluster_")]
        for c_col in cluster_cols:
            y = df[c_col].values
            exclude = [c for c in df.columns if c.startswith("cluster_") or c.startswith("x3")]
            X = df.drop(columns=exclude, errors="ignore")
            
            name, auc, _ = train_and_select_best(X, y)
            results.append({"Dataset": ds_name, "Cluster": c_col, "BestModel": name or "NA", "AUC": auc})

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Unassisted classification complete. Saved to {OUTPUT_FILE}")
