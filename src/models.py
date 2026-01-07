from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_base_models():
    """
    Initializes the diverse regressors defined in the project requirements.
    Each model is wrapped in a pipeline to ensure scaling is handled safely.
    """
    
    # 1. Elastic Net: Linear with L1/L2. Requires scaling[cite: 56].
    en_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
    ])
    
    # 2. Random Forest: Non-linear, robust to outliers. Scaling optional[cite: 56].
    rf_model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=10, 
        min_samples_leaf=5, 
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Gradient Boosting: Captures complex sequential patterns[cite: 57].
    gbrt_model = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
    
    return {
        "ElasticNet": en_pipeline,
        "RandomForest": rf_model,
        "GBRT": gbrt_model
    }

def get_null_baseline(y_train):
    """
    Returns the Naive (0) mean return from the training set.
    Used as a benchmark for MAE and RMSE.
    """
    return y_train.mean()