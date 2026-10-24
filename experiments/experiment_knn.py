from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .base_experiment import BaseExperiment, feature_importances
from .utils import safe_rmsle, save_model, FeatureNamePreserver
from experiments.utils import save_state, load_state
import logging
import pandas as pd
import tensorflow as tf
import time
import numpy as np

# Suppress specific TensorFlow warnings if TensorFlow is used elsewhere
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Unregister existing TensorFlow plugins if registered
try:
    tf.experimental.unregister_fft_plugin('cuFFT')
    logging.info("Successfully unregistered existing TensorFlow plugins in KNN.")
except AttributeError:
    logging.info("No existing TensorFlow plugins to unregister in KNN.")

# Re-register TensorFlow plugins
try:
    tf.experimental.register_fft_plugin('cuFFT')
    logging.info("Successfully registered TensorFlow plugins in KNN.")
except AttributeError as e:
    logging.error(f"TensorFlow plugin registration methods not found in KNN: {e}")
except Exception as e:
    logging.error(f"Failed to register TensorFlow plugins in KNN: {e}")

logger = logging.getLogger(__name__)

class KNNExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "KNN"
        
    def create_model(self):
        return Pipeline([
            ('names', FeatureNamePreserver()),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor())
        ])
    
    def run(self):
        try:
            start_time = time.time()
            
            X, y = self.load_data()
            
            # Filter columns using feature_importances from base_experiment
            important_features = [feat[0] for feat in feature_importances if feat[1] > 0.001]
            X = X[important_features]
            self.logger.info(f"Using {len(important_features)} features based on importance: {important_features}")
            
            self.logger.info(f"Loaded data: X shape={X.shape}, y shape={y.shape}")
            
            # Ensure target values are non-negative
            if (y < 0).any():
                raise ValueError("Target values contain negative values, which are not allowed for mean_squared_log_error.")
            
            # Define parameter distributions
            param_distributions = {
                'knn__n_neighbors': list(range(1, 10, 2)),
                'knn__weights': ['uniform', 'distance'],
                'knn__p': [1, 2]  # 1: Manhattan, 2: Euclidean
            }
            
            # Use Repeated K-Fold
            rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
            
            # Randomized search over parameter space
            self.logger.info("Starting parameter search for KNN...")
            search = RandomizedSearchCV(
                self.create_model(),
                param_distributions=param_distributions,
                n_iter=3,
                scoring='neg_mean_squared_log_error',
                cv=rkf,
                verbose=2,
                random_state=42,
                n_jobs=-1
            )
            search.fit(X, y)
            self.logger.info("KNN parameter search completed.")

            end_time = time.time()
            self.logger.info(f"KNN hyperparameter tuning took ~{end_time - start_time:.2f} seconds.")
            
            # Best model
            best_pipeline = search.best_estimator_
            best_score_msle = -search.best_score_
            msle = best_score_msle if best_score_msle > 0 else 0.0
            final_rmsle = np.sqrt(msle)
            self.logger.info(f"Best KNN RMSLE: {final_rmsle:.4f}")
            
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
            
            self.logger.info(f"Saving KNN model with score: {final_rmsle:.4f}")
            save_model(model_info, self.name, self.output_dir)
            
            self.logger.info("KNN model saved successfully")
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

def run_knn(output_dir):
    experiment = KNNExperiment(output_dir)
    model_info = experiment.run()
    
    # Load existing state
    state = load_state(output_dir)
    
    if model_info['status'] == 'success':
        # Append as a dict with 'name' and 'score'
        state['completed'].append({
            'name': model_info['name'],
            'score': model_info['score']
        })
    elif model_info['status'] == 'failed':
        state['failed'].append((model_info['name'], model_info['error']))
    else:
        state['crashed'].append(model_info['name'])
    
    # Save updated state
    save_state(output_dir, state['completed'], state['failed'], state['crashed'])
    
    return model_info

if __name__ == "__main__":
    run_knn('output')
