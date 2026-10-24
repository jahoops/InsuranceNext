from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from .base_experiment import BaseExperiment
from .utils import safe_rmsle, save_model, FeatureNamePreserver
import logging
import numpy as np
import pandas as pd

class GradientBoostingExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "GradientBoosting"
        
    def create_model(self):
        return Pipeline([
            ('names', FeatureNamePreserver()),
            ('scaler', RobustScaler(quantile_range=(5, 95))),
            ('gbr', GradientBoostingRegressor(random_state=42))
        ])
    
    def run(self):
        try:
            X, y = self.load_data()
            self.logger.info(f"Loaded data: X shape={X.shape}, y shape={y.shape}")
            
            # Define parameter distributions
            param_distributions = {
                'gbr__n_estimators': [100, 200, 300],
                'gbr__learning_rate': [0.01, 0.05, 0.1],
                'gbr__max_depth': [3, 5, 7],
                'gbr__min_samples_split': [2, 5, 10],
                'gbr__min_samples_leaf': [1, 2, 4],
                'gbr__max_features': ['auto', 'sqrt', 'log2']
            }
            
            # Use Repeated K-Fold
            rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
            
            # Randomized search over parameter space
            search = RandomizedSearchCV(
                self.create_model(),
                param_distributions=param_distributions,
                n_iter=20,
                scoring='neg_mean_squared_log_error',
                cv=rkf,
                verbose=2,
                random_state=42,
                n_jobs=-1
            )
            search.fit(X, y)
            
            # Best model
            best_pipeline = search.best_estimator_
            best_score_msle = -search.best_score_
            msle = best_score_msle if best_score_msle > 0 else 0.0
            final_rmsle = np.sqrt(msle)
            self.logger.info(f"Best GradientBoosting RMSLE: {final_rmsle:.4f}")
            
            model_info = {
                'name': self.name,
                'model': best_pipeline,
                'features': X.columns.tolist(),
                'score': final_rmsle,
                'status': 'success',
                'error': None,
                'version': str(self.output_dir),
                'sample_size': len(X)
            }
            
            self.logger.info(f"Saving GradientBoosting model with score: {final_rmsle:.4f}")
            save_model(model_info, self.name, self.output_dir)
            
            self.logger.info("GradientBoosting model saved successfully")
            return model_info
            
        except Exception as e:
            error_msg = f"{self.name} failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'model': None,
                'features': [],
                'score': float('inf'),
                'status': 'failed',
                'error': error_msg
            }

def run_gradient_boosting(output_dir):
    experiment = GradientBoostingExperiment(output_dir)
    return experiment.run()

if __name__ == "__main__":
    run_gradient_boosting('output')