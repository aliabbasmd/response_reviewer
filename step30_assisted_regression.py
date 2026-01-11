import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm

# --- CONFIG ---
DATA_DIR = "generated_data_with_clusters"
TOP5_FEATURES_FILE = "top5_features_per_dataset.csv"
OUTPUT_DIR = "llm_assisted_regression_results"
OUTPUT_FILE = "llm_assisted_regression_results.csv"
RANDOM_SEED = 42

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

REG_MODELS = {
    "LinearReg": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
    "GradientBoost": GradientBoostingRegressor(random_state=RANDOM_SEED)
}

# Load the top 5 features dataset
df_top5 = pd.read_csv(TOP5_FEATURES_FILE)
results = []

for index, row in df_top5.iterrows():
    dataset_name = row["Dataset"]
    top5_features = [f.strip() for f in row["Top5_Features"].split(",")]
    
    file_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    if not os.path.exists(file_path):
        continue
    
    print(f"📊 Regression on: {dataset_name}")
    df = pd.read_csv(file_path)
    df["x3"] = pd.to_numeric(df["x3"], errors="coerce")
    df.dropna(subset=["x3"], inplace=True)

    # Preprocessing
    # Important: Remove target from features list if it was mistakenly selected
    top5_features = [f for f in top5_features if f != "x3" and f in df.columns]
    
    X = df[top5_features]
    y = df["x3"]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    # Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    best_m, best_r2, best_mae, best_eq = None, -np.inf, np.inf, ""

    for name, reg_model in REG_MODELS.items():
        pipe = Pipeline([("prep", preprocessor), ("model", reg_model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Logic for selecting best model (Primary metric: R2)
        if r2 > best_r2:
            best_r2, best_mae, best_m = r2, mae, name
            if name == "LinearReg":
                # Extract equation logic
                try:
                    # Transform X_train for OLS to get coefficients
                    X_tr_proc = preprocessor.transform(X_train)
                    if hasattr(X_tr_proc, "toarray"): X_tr_proc = X_tr_proc.toarray()
                    X_tr_sm = sm.add_constant(X_tr_proc)
                    ols = sm.OLS(y_train, X_tr_sm).fit()
                    best_eq = f"y = {ols.params[0]:.3f} + sum(beta*x)"
                except:
                    best_eq = "Linear eq available in detailed log"
            else:
                best_eq = "Non-linear"

    results.append({
        "Dataset": dataset_name,
        "BestModel": best_m,
        "R2": round(best_r2, 4),
        "MAE": round(best_mae, 4),
        "Equation": best_eq
    })

# Save results
pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILE), index=False)
print(f"✅ Regression results saved to {OUTPUT_DIR}/{OUTPUT_FILE}")
