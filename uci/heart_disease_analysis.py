import pandas as pd
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from ucimlrepo import fetch_ucirepo
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, roc_curve, auc, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from transformers import (BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, 
                          DistilBertTokenizer, DistilBertModel, T5Tokenizer, 
                          T5EncoderModel, AutoTokenizer, AutoModel, XLNetModel, XLNetTokenizer)

# 1. DATA INGESTION & CLINICAL MAPPING
print("Fetching dataset...")
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

X['sex'] = X['sex'].map({1: 'male', 0: 'female'})
X['cp'] = X['cp'].map({1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal pain', 4: 'asymptomatic'})
X['restecg'] = X['restecg'].map({0: 'normal', 1: 'ST-T wave abnormality', 2: 'left ventricular hypertrophy'})
X['exang'] = X['exang'].map({1: 'exercise induced angina present', 0: 'no exercise induced angina'})
X['thal'] = X['thal'].map({3: 'normal', 6: 'fixed defect', 7: 'reversible defect'})
X['outcome'] = y['num'].map(lambda x: 'presence of heart disease' if x > 0 else 'absence of heart disease')

# 2. FEATURE STRINGIFICATION
def create_sentence_from_row(row):
    sentence_parts = []
    for column, value in row.items():
        if column in ['outcome', 'ca']: continue
        if value == 0: sentence_parts.append(f"no {column}")
        elif value == 1: sentence_parts.append(column)
        else: sentence_parts.append(f"{value} {column}")
    return ' '.join(sentence_parts)

X['combined'] = X.apply(create_sentence_from_row, axis=1)

# 3. TRANSFORMER EMBEDDING GENERATION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_embeddings(tokenizer, model, texts, device):
    model.eval()
    embeddings_list = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            if isinstance(model, T5EncoderModel):
                outputs = model.encoder(**inputs)
            else:
                outputs = model(**inputs)
        embeddings_list.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings_list)

print("Generating T5 Embeddings...")
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5EncoderModel.from_pretrained('t5-small').to(device)
t5_embeddings = get_embeddings(t5_tokenizer, t5_model, X['combined'].tolist(), device)

# 4. CLUSTERING & EVALUATION
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(t5_embeddings)
X['t5_cluster'] = kmeans.labels_
sil_score = silhouette_score(t5_embeddings, kmeans.labels_)
print(f"T5 Clustering Silhouette Score: {sil_score:.4f}")

# 5. SUPERVISED LEARNING & SHAP
y_binary = y['num'].map(lambda x: 1 if x > 0 else 0)
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), categorical_features)
])

clf_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['outcome', 'combined', 't5_cluster']), y_binary, test_size=0.2, random_state=42)
clf_pipeline.fit(X_train, y_train)

print("Analysis complete. Script finished.")
