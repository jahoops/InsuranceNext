from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold  # Added RepeatedKFold import
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from .base_experiment import BaseExperiment
from .utils import safe_rmsle, save_model, FeatureNamePreserver
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

# Suppress specific TensorFlow warnings if TensorFlow is used elsewhere
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Unregister existing TensorFlow plugins if registered
try:
    tf.experimental.unregister_fft_plugin('cuFFT')
    logging.info("Successfully unregistered existing TensorFlow plugins in LightGBM.")
except AttributeError:
    logging.info("No existing TensorFlow plugins to unregister in LightGBM.")

# Re-register TensorFlow plugins
try:
    tf.experimental.register_fft_plugin('cuFFT')
    logging.info("Successfully registered TensorFlow plugins in LightGBM.")
except AttributeError as e:
    logging.error(f"TensorFlow plugin registration methods not found in LightGBM: {e}")
except Exception as e:
    logging.error(f"Failed to register TensorFlow plugins in LightGBM: {e}")

class LightGBMExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "LightGBM"
        
    def create_model(self):
        return Pipeline([
            ('names', FeatureNamePreserver()),
            ('scaler', RobustScaler(quantile_range=(5, 95))),
            ('lgbm', LGBMRegressor(
                objective='regression',
                random_state=42,
                n_jobs=-1,
                device='gpu'  # Ensure GPU usage
            ))
        ])
    
    def run(self):
        try:
            X, y = self.load_data()
            self.logger.info(f"Loaded data: X shape={X.shape}, y shape={y.shape}")

            # Remove 'id' column if present
            if 'id' in X.columns:
                X = X.drop(columns=['id'])
                self.logger.info("'id' column removed from features")
            
            # Define parameter distributions
            param_distributions = {
                'lgbm__n_estimators': [100, 200, 300, 500],
                'lgbm__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'lgbm__num_leaves': [31, 50, 70, 100],
                'lgbm__max_depth': [-1, 10, 20, 30],
                'lgbm__min_child_samples': [20, 30, 40],
                'lgbm__subsample': [0.6, 0.8, 1.0],
                'lgbm__colsample_bytree': [0.6, 0.8, 1.0]
            }
            
            # Use Repeated K-Fold
            rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
            
            # Randomized search over parameter space
            search = RandomizedSearchCV(
                self.create_model(),
                param_distributions=param_distributions,
                n_iter=30,
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
            self.logger.info(f"Best LightGBM RMSLE: {final_rmsle:.4f}")
            
            feature_importance = dict(zip(
                X.columns,
                best_pipeline.named_steps['lgbm'].feature_importances_
            ))

            model_info = {
                'name': self.name,
                'model': best_pipeline,
                'features': X.columns.tolist(),
                'score': final_rmsle,
                'feature_importance': feature_importance,
                'status': 'success',
                'error': None,
                'version': str(self.output_dir),
                'sample_size': len(X)
            }
            
            self.logger.info(f"Saving LightGBM model with score: {final_rmsle:.4f}")
            save_model(model_info, self.name, self.output_dir)
            
            self.logger.info("LightGBM model saved successfully")
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

def run_lightgbm(output_dir):
    experiment = LightGBMExperiment(output_dir)
    return experiment.run()

if __name__ == "__main__":
    run_lightgbm('output')
