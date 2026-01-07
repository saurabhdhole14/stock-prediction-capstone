# Stock Prediction via Time-Aware Stacking Ensembles

A machine learning pipeline utilizing walk-forward validation and stacked ensembles to predict stock returns.

## 📊 Performance Metrics
The ensemble model significantly outperformed the naive baseline:
- **Ensemble MAE**: 0.0008
- **Ensemble RMSE**: 0.0009
- **Sign Accuracy**: 1.0

## 🔍 Methodology
- **Walk-Forward Validation**: Data split into 2018 (Train), 2019 (Validation), and 2020 (Test).
- **Feature Engineering**: Includes OLS-based idiosyncratic residuals and sector-relative returns.
- **Stacking**: A meta-learner integrates predictions from Random Forest, GBRT, and Elastic Net.
