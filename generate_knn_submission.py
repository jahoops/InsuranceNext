import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_knn_submission():
    try:
        # Specific path to the best KNN model
        model_path = Path('output/v1/best_KNN_model.pkl')
        logger.info(f"Loading KNN model from {model_path}")
        
        model_info = joblib.load(model_path)
        model = model_info['model']
        features = model_info['features']
        
        logger.info(f"Original model score: {model_info['score']:.4f}")
        
        # Load test data
        test_data = pd.read_csv('output/cleaned_test.csv')
        logger.info(f"Test data shape: {test_data.shape}")
        
        # Ensure we have all required features
        X_test = test_data[features]
        logger.info(f"Using {len(features)} features")
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = model.predict(X_test)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_data['id'],
            'PremiumAmount': predictions
        })
        
        # Save submission
        submission_file = 'output/knn_only_submission.csv'
        submission.to_csv(submission_file, index=False)
        
        logger.info(f"\nSubmission saved to: {submission_file}")
        logger.info(f"Predictions range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        logger.info(f"Mean prediction: ${predictions.mean():.2f}")
        logger.info(f"Median prediction: ${np.median(predictions):.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating KNN submission: {str(e)}")
        return False

if __name__ == "__main__":
    generate_knn_submission()
