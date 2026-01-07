import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

class TimeAwareStacker:
    def __init__(self):
        # Base learners as required by the project [cite: 56, 57]
        self.base_models = {
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10),
            'gbrt': GradientBoostingRegressor(n_estimators=100)
        }
        # Meta-learner to combine base predictions [cite: 62]
        self.meta_learner = Lasso(alpha=0.0001, positive=True) 
        self.scaler = StandardScaler()

    def get_oof_predictions(self, X_val, y_val):
        """
        Creates the meta-matrix Z using a rolling TimeSeriesSplit.
        """
        tscv = TimeSeriesSplit(n_splits=5)
        # Z will hold predictions from each base model [cite: 62]
        Z = np.zeros((len(X_val), len(self.base_models)))
        
        for train_idx, val_idx in tscv.split(X_val):
            X_tr, X_holdout = X_val.iloc[train_idx], X_val.iloc[val_idx]
            y_tr = y_val.iloc[train_idx]
            
            for i, (name, model) in enumerate(self.base_models.items()):
                # Train on past blocks, predict the next block [cite: 61]
                model.fit(X_tr, y_tr)
                Z[val_idx, i] = model.predict(X_holdout)
                
        # Remove the first training block (which has no OOF predictions)
        mask = ~np.all(Z == 0, axis=1)
        return Z[mask], y_val.iloc[mask]

    def train_meta_learner(self, Z, y_val_masked):
        """
        Trains the meta-learner g(Z) -> r_{i,t+1}[cite: 62].
        """
        self.meta_learner.fit(Z, y_val_masked)
        return self.meta_learner

    def fit_base_on_full_data(self, X_train_val, y_train_val):
        """
        Refit each base model on combined Train + Validation before Test[cite: 65, 114].
        """
        for name, model in self.base_models.items():
            model.fit(X_train_val, y_train_val)

    def predict(self, X_test):
        """
        Feed Test predictions through the frozen stacker[cite: 65].
        """
        # Generate predictions from each base model
        base_preds = []
        for name, model in self.base_models.items():
            base_preds.append(model.predict(X_test))
            
        Z_test = np.column_stack(base_preds)
        # Final ensemble prediction [cite: 69]
        return self.meta_learner.predict(Z_test)