# Frontiers: LLM-Assisted Informatics Research Pipeline

This repository contains a complete 39-step computational pipeline designed to evaluate the efficacy of Large Language Models (LLMs) in interpreting complex mathematical structures.

## 🚀 Project Overview
This study explores the intersection of semantic AI logic and traditional machine learning. We use LLM-generated embeddings to cluster synthetic mathematical datasets and then validate those clusters against traditional statistical baselines (Logistic and Linear Regression).

## 📂 Key Pipeline Stages
1. **Data Generation:** Synthetic creation of quadratic, cubic, exponential, and linear datasets.
2. **LLM Processing:** Generating embeddings using models like BERT, Llama, and RoBERTa.
3. **Clustering & Similarity:** K-Means clustering and evaluation via Silhouette scores and Rand Indices.
4. **Predictive Validation:** - **Assisted Models:** ML models utilizing LLM clusters/top features.
   - **Unassisted Baselines:** Traditional models running on raw numerical data.
5. **Statistical Diagnostics:** Breusch-Pagan tests, Subject-Variable Ratio (SVR) analysis, and residual diagnostics.
6. **Performance Delta:** Paired t-tests and Wilcoxon signed-rank tests to measure the "LLM Lift."

## 🛠 Installation
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## ✍️ Author
**Ali Abbas** - GitHub: [aliabbasmd](https://github.com/aliabbasmd)
