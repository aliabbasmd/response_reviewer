import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap

# --- CONFIG ---
DATA_DIR = "generated_data_with_clusters"
OUTPUT_DIR = "numeric_x3_results_extended"
UNASSISTED_FILE = "unassisted_model_results_extended.csv"
RANDOM_SEED = 42

REG_MODELS = {
    "LinearReg": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
    "GradientBoost": GradientBoostingRegressor(random_state=RANDOM_SEED)
}

def build_preprocessor(numeric_cols, cat_cols):
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

def evaluate_regression_models(X, y, numeric_cols, cat_cols):
    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    
    best_r2 = -np.inf
    results = {}

    for name, model in REG_MODELS.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        if r2 > best_r2:
            best_r2, results = r2, {"name": name, "r2": r2, "rmse": rmse, "mae": mae, "pipe": pipe}
            
    return results

def compute_shap_top_features(pipe, X, numeric_cols, cat_cols):
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    X_enc = prep.transform(X)
    if hasattr(X_enc, "toarray"): X_enc = X_enc.toarray()
    
    model_type = type(model).__name__
    if "Forest" in model_type or "Boost" in model_type:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_enc, check_additivity=False)
    else:
        explainer = shap.LinearExplainer(model, X_enc)
        shap_vals = explainer.shap_values(X_enc)
    
    mean_shap = np.abs(shap_vals).mean(axis=0)
    # Simplified feature name retrieval for the script
    return [f"feat_{i}" for i in np.argsort(mean_shap)[::-1][:5]]

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    csv_files = glob(os.path.join(DATA_DIR, "*.csv"))
    all_results = []

    for path in csv_files:
        ds_name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path).dropna(subset=["x3"])
        
        cat_cols = [c for c in df.columns if c.startswith("cat")]
        for c in cat_cols: df[c] = df[c].astype("category")
        numeric_cols = [c for c in ["x1", "x2"] if c in df.columns]

        X, y = df[numeric_cols + cat_cols], df["x3"]
        res = evaluate_regression_models(X, y, numeric_cols, cat_cols)
        
        if res:
            print(f"Dataset: {ds_name} | Best: {res['name']} | R2: {res['r2']:.3f}")
            top_feats = compute_shap_top_features(res['pipe'], X, numeric_cols, cat_cols)
            all_results.append({
                "Dataset": ds_name, "BestModel": res['name'], 
                "R2": res['r2'], "RMSE": res['rmse'], "MAE": res['mae'],
                "TopFeatures": ", ".join(top_feats)
            })

    pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_DIR, UNASSISTED_FILE), index=False)
    print(f"✅ Extended results saved to {OUTPUT_DIR}/{UNASSISTED_FILE}")
