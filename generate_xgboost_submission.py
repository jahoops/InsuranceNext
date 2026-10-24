import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from experiments.base_experiment import feature_importances  # Add this import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_xgboost_submission():
    try:
        # Try different possible model paths
        possible_paths = [
            Path('output/v1/best_XGBoost_model.pkl'),
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
                
        if model_path is None:
            raise FileNotFoundError(f"Could not find XGBoost model in any of: {possible_paths}")
            
        logger.info(f"Loading XGBoost model from {model_path}")
        model_info = joblib.load(model_path)
        
        # Validate model info contents
        required_keys = ['model', 'features', 'score']
        missing_keys = [key for key in required_keys if key not in model_info]
        if missing_keys:
            raise KeyError(f"Model info is missing required keys: {missing_keys}")
        
        logger.info(f"Original model score: {model_info['score']:.4f}")
        logger.info(f"Number of features: {len(model_info['features'])}")
        logger.info(f"Features used: {', '.join(model_info['features'])}")
        
        model = model_info['model']
        features = model_info['features']
        
        # Load test data
        test_data = pd.read_csv('output/cleaned_test.csv')
        logger.info(f"Test data shape: {test_data.shape}")
        
        # Use exactly the same features as training
        X_test = test_data[features]  # Use features from model_info instead of feature_importances
        logger.info(f"Using {len(features)} features from trained model")
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = model.predict(X_test)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_data['id'],
            'Premium Amount': predictions
        })
        
        # Save submission
        submission_file = 'output/xgboost_submission.csv'
        submission.to_csv(submission_file, index=False)
        
        logger.info(f"\nSubmission saved to: {submission_file}")
        logger.info(f"Predictions range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        logger.info(f"Mean prediction: ${predictions.mean():.2f}")
        logger.info(f"Median prediction: ${np.median(predictions):.2f}")
        
        return True
        
    except Exception as e:
        # Update error message to reference XGBoost instead of KNN
        logger.error(f"Error generating XGBoost submission: {str(e)}")
        return False

if __name__ == "__main__":
    generate_xgboost_submission()
