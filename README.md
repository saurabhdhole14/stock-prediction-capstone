# Stock Prediction via Time-Aware Stacking Ensembles

This project implements a machine learning pipeline for stock return prediction using a walk-forward validation framework and an ensemble stacking architecture.

## 📊 Performance Metrics (2020-Q1 Test Period)
The ensemble model significantly outperformed the naive baseline and individual base learners:

| Model | MAE | RMSE | Sign Accuracy |
| :--- | :--- | :--- | :--- |
| **Ensemble** | **0.0008** | **0.0009** | **1.0** |
| Random Forest | 0.0010 | 0.0010 | 1.0 |
| GBRT | 0.0020 | 0.0023 | 1.0 |
| Naive (Baseline) | 0.0130 | 0.0141 | 0.0 |

## 🔍 Key Findings
### Feature Ablation Study
The inclusion of market and sector-specific context improved directional predictive power over using own-stock history alone:
- **Own history only**: 0.53 Sign Accuracy
- **Own + Market + Sector**: 0.55 Sign Accuracy

## 🛠️ Methodology
1. **Walk-Forward Validation**: Data split into 2018 (Train), 2019 (Validation), and 2020 (Test) to simulate real-world conditions.
2. **Feature Engineering**: Includes lagged returns, rolling volatility, and OLS-based idiosyncratic residuals.
3. **Stacking Ensemble**: A meta-learner (ElasticNet) integrates predictions from Random Forest and Gradient Boosted Trees.

## 📂 Project Structure
- `src/`: Core logic for feature engineering and model stacking.
- `main.py`: Full execution pipeline.
- `artifacts/`: Generated metrics tables and performance visualizations.
