import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """
    Calculates the three core metrics required for the project report.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Sign Accuracy: Pr[sign(hat{r}) == sign(r)]
    # Measures how often the model correctly predicts the direction
    correct_sign = (np.sign(y_true) == np.sign(y_pred)).sum()
    sign_acc = correct_sign / len(y_true)
    
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "SignAcc": round(sign_acc, 2)
    }

def generate_metrics_table(test_results_df):
    """
    Expects a DataFrame with actual returns and predictions from each model.
    Columns: ['actual', 'naive', 'elastic_net', 'rf', 'gbrt', 'ensemble']
    """
    models = ['naive', 'elastic_net', 'rf', 'gbrt', 'ensemble']
    summary = []
    
    for model in models:
        m = calculate_metrics(test_results_df['actual'], test_results_df[model])
        m['Model'] = model
        summary.append(m)
    
    return pd.DataFrame(summary)[['Model', 'MAE', 'RMSE', 'SignAcc']]