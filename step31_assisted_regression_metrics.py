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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm

# --- CONFIG ---
DATA_DIR = "generated_data_with_clusters"
OUTPUT_DIR = "llm_assisted_linear_results_with_metrics"
TOP5_FEATURES_FILE = "top5_features_per_dataset.csv"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "llm_assisted_regression_results.csv")
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

REG_MODELS = {
    "LinearReg": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
    "GradientBoost": GradientBoostingRegressor(random_state=RANDOM_SEED)
}

df_top5 = pd.read_csv(TOP5_FEATURES_FILE)
results = []

for index, row in df_top5.iterrows():
    dataset_name = row["Dataset"]
    top5_features = [f.strip() for f in row["Top5_Features"].split(",")]
    model_source = row["Model"]

    file_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    if not os.path.exists(file_path):
        continue
    
    print(f"📈 Processing {dataset_name} (Source: {model_source})")
    df = pd.read_csv(file_path)
    df["x3"] = pd.to_numeric(df["x3"], errors="coerce")
    df.dropna(subset=["x3"], inplace=True)

    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in df.columns if col not in numeric_features]

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    df_proc = preprocessor.fit_transform(df)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    feature_names = numeric_features + list(cat_encoder.get_feature_names_out(categorical_features))

    # Match top5 to encoded features
    selected_features = [f for f in feature_names if any(orig in f for orig in top5_features)]
    
    if not selected_features:
        continue

    X = pd.DataFrame(df_proc, columns=feature_names)[selected_features]
    y = df["x3"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    best_m, best_r2, best_rmse, best_mae, best_eq = None, -np.inf, np.inf, np.inf, None

    for name, reg_model in REG_MODELS.items():
        model = reg_model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        if r2 > best_r2:
            best_r2, best_rmse, best_mae, best_m = r2, rmse, mae, name
            if name == "LinearReg":
                X_sm = sm.add_constant(X_train)
                ols = sm.OLS(y_train, X_sm).fit()
                best_eq = f"y = {ols.params[0]:.4f} + sum({ols.params[1:].mean():.4f}*features)"
            else:
                best_eq = "Non-linear"

    results.append({
        "Dataset": dataset_name,
        "BestModel": best_m,
        "R2": round(best_r2, 4),
        "RMSE": round(best_rmse, 4),
        "MAE": round(best_mae, 4),
        "RegressionEquation": best_eq
    })

pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
print(f"✅ Full results saved to {OUTPUT_FILE}")
