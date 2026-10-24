import pandas as pd
import numpy as np
import json
import joblib
import logging
import warnings
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import UserWarning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter the specific warning
warnings.filterwarnings('ignore', category=UserWarning, 
                       message='X has feature names, but .* was fitted without feature names')

def get_model_accuracies_per_record(model_info, X_train, y_train):
    """Calculate how accurate the model was on each training record"""
    model = model_info['model']
    features = model_info['features']
    
    # Ensure X_train has exactly the features the model expects
    X_train_subset = X_train[features].copy()
    
    # If model is a Pipeline, get the final estimator
    if hasattr(model, 'steps'):
        final_estimator = model.steps[-1][1]
        if hasattr(final_estimator, 'feature_names_in_'):
            # Ensure feature order matches
            X_train_subset = X_train_subset[final_estimator.feature_names_in_]
    elif hasattr(model, 'feature_names_in_'):
        # Ensure feature order matches
        X_train_subset = X_train_subset[model.feature_names_in_]
    
    train_preds = model.predict(X_train_subset)
    errors = np.abs(train_preds - y_train) / y_train
    return errors, X_train_subset.columns

def get_model_path(base_path: str) -> Path:
    """Resolve model path, checking both with and without output prefix"""
    # Try direct path first
    path = Path(base_path)
    if (path.exists()):
        return path
    
    # Try with output prefix
    path = Path('output') / base_path
    if (path.exists()):
        return path
    
    # Try without output prefix
    path = Path(base_path.replace('output/', '', 1))
    if (path.exists()):
        return path
        
    raise FileNotFoundError(f"Could not find model at any of these locations: {base_path}, output/{base_path}")

def generate_nearest_match_submission():
    try:
        # Load training data for similarity matching
        train_data = pd.read_csv('output/cleaned_train.csv')
        X_train = train_data.drop(columns=['PremiumAmount', 'id'])  # Also drop id
        y_train = train_data['PremiumAmount']
        
        # Load test data (don't try to drop PremiumAmount)
        test_data = pd.read_csv('output/cleaned_test.csv')
        X_test = test_data.drop(columns=['id'])  # Only drop id
        
        logger.info(f"Training features: {X_train.shape[1]}")
        logger.info(f"Test features: {X_test.shape[1]}")
        
        # Verify feature alignment
        missing_cols = set(X_train.columns) - set(X_test.columns)
        extra_cols = set(X_test.columns) - set(X_train.columns)
        
        if missing_cols or extra_cols:
            raise ValueError(f"Feature mismatch! Missing: {missing_cols}, Extra: {extra_cols}")
        
        # Ensure column order matches
        X_test = X_test[X_train.columns]
        
        # Scale features for better similarity matching
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize nearest neighbors
        nn = NearestNeighbors(n_neighbors=5, algorithm='auto', n_jobs=-1)
        nn.fit(X_train_scaled)
        
        # Load models and calculate their accuracies on training data
        with open('output/best_models_config.json', 'r') as f:
            models_config = json.load(f)
        
        model_errors = {}
        models = {}
        
        # Load each model and calculate its errors on training data
        for model_name, config in models_config.items():
            if model_name == 'KNN':  # Skip KNN as it's already similarity-based
                continue
                
            logger.info(f"Loading {model_name} and calculating training errors...")
            try:
                model_path = get_model_path(config['path'])
                logger.info(f"Found model at: {model_path}")
                
                model_info = joblib.load(model_path)
                models[model_name] = model_info
                errors, feature_names = get_model_accuracies_per_record(
                    model_info, X_train, y_train
                )
                model_errors[model_name] = errors
                logger.info(f"Using {len(feature_names)} features for {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name} model: {e}")
                continue
        
        if not models:
            raise ValueError("No models could be loaded successfully")
        
        # Generate predictions
        final_predictions = []
        model_usage_counts = {name: 0 for name in models}
        
        # Process test data in batches
        batch_size = 1000
        n_batches = len(X_test) // batch_size + 1
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(X_test))
            
            if batch % 10 == 0:
                logger.info(f"Processing batch {batch+1}/{n_batches}...")
            
            # Get current batch
            X_batch_scaled = X_test_scaled[start_idx:end_idx]
            
            # Find nearest training samples for this batch
            distances, indices = nn.kneighbors(X_batch_scaled)
            
            # For each test record in batch
            batch_predictions = []
            for i in range(len(X_batch_scaled)):
                similar_indices = indices[i]
                similar_distances = distances[i]
                
                # Calculate weighted accuracy of each model on similar samples
                model_scores = {}
                for name, errors in model_errors.items():
                    # Weight errors by inverse distance
                    weighted_errors = np.average(
                        errors[similar_indices],
                        weights=1/(similar_distances + 1e-8)
                    )
                    model_scores[name] = weighted_errors
                
                # Select best model for this record
                best_model_name = min(model_scores.items(), key=lambda x: x[1])[0]
                model_usage_counts[best_model_name] += 1
                
                # Get prediction from best model
                best_model_info = models[best_model_name]
                features = best_model_info['features']
                X_pred = X_test[features].iloc[[start_idx + i]]
                
                # Ensure feature order matches what model expects
                if hasattr(best_model_info['model'], 'feature_names_in_'):
                    X_pred = X_pred[best_model_info['model'].feature_names_in_]
                elif hasattr(best_model_info['model'], 'steps'):
                    final_estimator = best_model_info['model'].steps[-1][1]
                    if hasattr(final_estimator, 'feature_names_in_'):
                        X_pred = X_pred[final_estimator.feature_names_in_]
                
                pred = best_model_info['model'].predict(X_pred)
                batch_predictions.append(pred[0])
            
            final_predictions.extend(batch_predictions)
        
        final_predictions = np.array(final_predictions)
        final_predictions = np.maximum(final_predictions, 0)
        
        # Log model usage statistics
        logger.info("\nModel Usage Statistics:")
        total_records = sum(model_usage_counts.values())
        for model_name, count in model_usage_counts.items():
            percentage = (count / total_records) * 100
            logger.info(f"{model_name}: {count} records ({percentage:.1f}%)")
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_data['id'],
            'PremiumAmount': final_predictions
        })
        
        output_path = 'output/nearest_match_ensemble_submission.csv'
        submission.to_csv(output_path, index=False)
        
        logger.info(f"\nSubmission saved to {output_path}")
        logger.info(f"Predictions range: {final_predictions.min():.2f} - {final_predictions.max():.2f}")
        logger.info(f"Mean prediction: {final_predictions.mean():.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating submission: {str(e)}")
        return False

if __name__ == "__main__":
    generate_nearest_match_submission()
