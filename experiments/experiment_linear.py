from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import cross_val_score, KFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error  # Add this import
from sklearn.utils import gen_batches
from .base_experiment import BaseExperiment, feature_importances  # Add this import
from tqdm import tqdm
import numpy as np
import time
from datetime import timedelta
import pandas as pd
from .utils import FeatureNamePreserver

class LinearExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "Linear"
        
    def create_model(self):
        X, y = self.load_data()  # Load data once at the start
        start_time = time.time()
        
        # Ensure target values are non-negative
        if (y < 0).any():
            raise ValueError("Target values contain negative values, which are not allowed for mean_squared_log_error.")
        
        # Add sampling for large datasets
        if len(X) > 50000:
            self.logger.info("Large dataset detected, using stratified sampling")
            X_sample, _, y_sample, _ = train_test_split(
                X, y, 
                train_size=50000,  # Use larger sample than SVR but still manageable
                random_state=42,
                stratify=pd.qcut(y, q=10, labels=False, duplicates='drop')
            )
            X, y = X_sample, y_sample
            self.logger.info(f"Sampled down to {len(X)} records")

        # Reduce polynomial complexity
        poly_degree = 1 if len(X.columns) > 20 else 2
        self.logger.info(f"Using polynomial degree {poly_degree} for {len(X.columns)} features")
        
        models = {
            'elasticnet': ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=2000,
                tol=1e-4,
                warm_start=True,        # Add warm start for faster convergence
                selection='cyclic'      # Faster than 'random'
            ),
            'ridge': Ridge(
                alpha=0.1,
                max_iter=2000,
                tol=1e-4,
                solver='svd'            # Fastest solver for parallel processing
            ),
            'lasso': Lasso(
                alpha=0.001,
                max_iter=2000,
                tol=1e-4,
                selection='cyclic',    # Faster than 'random'
                warm_start=True        # Add warm start
            )
        }
        
        best_score = float('inf')
        best_model = None
        
        # Progress bar for models
        pbar_models = tqdm(models.items(), desc="Models", total=len(models))
        
        for name, model in pbar_models:
            model_start = time.time()
            self.logger.info(f"\nStarting {name} evaluation...")
            
            pipeline = Pipeline([
                ('names', FeatureNamePreserver()),
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(
                    degree=poly_degree, 
                    include_bias=False,
                    interaction_only=True  # Only use interaction terms, not full polynomial
                )),
                ('selector', SelectKBest(score_func=mutual_info_regression, k=min(20, X.shape[1]))),
                ('regressor', model)
            ])
            
            # Define parameter distributions for RandomizedSearchCV
            param_distributions = {
                'poly__degree': [1, 2, 3],
                'regressor__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9] if name == 'elasticnet' else [None]
            }
            
            # Use Repeated K-Fold
            rkf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Randomized search over parameter space
            search = RandomizedSearchCV(
                pipeline, param_distributions=param_distributions,
                n_iter=50, scoring='neg_mean_squared_log_error',
                cv=rkf, verbose=2, random_state=42, n_jobs=-1
            )
            
            # Fit the model
            search.fit(X, y)
            
            # Best model
            best_pipeline = search.best_estimator_
            best_score_msle = -search.best_score_
            msle = best_score_msle if best_score_msle > 0 else 0.0
            final_rmsle = np.sqrt(msle)
            self.logger.info(f"Best {name} RMSLE: {final_rmsle:.4f}")
            
            if final_rmsle < best_score:
                best_score = final_rmsle
                best_model = best_pipeline
                self.logger.info(f"New best model: {name}")
            
            # Update progress bar description with timing info
            elapsed = time.time() - start_time
            pbar_models.set_description(f"Models ({timedelta(seconds=int(elapsed))})")
        
        self.logger.info(f"\nSelected best model with score: {best_score:.4f}")
        total_time = time.time() - start_time
        self.logger.info(f"Total training time: {timedelta(seconds=int(total_time))}")
        
        self.logger.info("Fitting best model on full dataset...")
        try:
            # Try batch fitting first
            best_model = self.fit_model_in_batches(best_model, X, y)
        except Exception as e:
            self.logger.warning(f"Batch fitting failed: {e}, falling back to regular fit with sample")
            # Fall back to sampled fit if batch fitting fails
            if len(X) > 100000:
                X_final, _, y_final, _ = train_test_split(
                    X, y, 
                    train_size=100000,
                    random_state=42,
                    stratify=pd.qcut(y, q=10, labels=False, duplicates='drop')
                )
                self.logger.info(f"Using {len(X_final)} samples for final fit")
                best_model.fit(X_final, y_final)
            else:
                best_model.fit(X, y)
        
        return best_model

    def fit_model_in_batches(self, model, X, y, batch_size=10000):
        """Fit model in batches to handle large datasets"""
        self.logger.info(f"Fitting final model on {len(X)} samples in batches of {batch_size}")
        n_samples = X.shape[0]
        
        with tqdm(total=n_samples, desc="Final Fit Progress") as pbar:
            for batch_idx in gen_batches(n_samples, batch_size):
                X_batch = X.iloc[batch_idx]
                y_batch = y.iloc[batch_idx]
                
                # Partial fit for models that support it
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(X_batch, y_batch)
                else:
                    # Regular fit for current batch
                    model.fit(X_batch, y_batch)
                
                pbar.update(len(X_batch))
                
        return model

def run_linear(output_dir):
    experiment = LinearExperiment(output_dir)
    return experiment.run()
