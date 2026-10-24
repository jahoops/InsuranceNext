import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_svr_submission():
    try:
        # Load best SVR model from v6 (or whichever version has best SVR)
        model_path = Path('output/v1/best_SVR_model.pkl')
        logger.info(f"Loading SVR model from {model_path}")
        
        model_info = joblib.load(model_path)
        model = model_info['model']
        features = model_info['features']
        
        logger.info(f"Original model score: {model_info['score']:.4f}")
        logger.info(f"Sample size used: {model_info.get('sample_size', 'unknown')}")
        
        # Load test data
        test_data = pd.read_csv('output/cleaned_test.csv')
        logger.info(f"Test data shape: {test_data.shape}")
        
        # Verify features
        missing_features = set(features) - set(test_data.columns)
        if missing_features:
            raise ValueError(f"Missing features in test data: {missing_features}")
            
        X_test = test_data[features]
        logger.info(f"Using {len(features)} features")
        
        # Generate predictions in batches to manage memory
        batch_size = 5000
        predictions = []
        
        for i in range(0, len(X_test), batch_size):
            batch = X_test.iloc[i:i+batch_size]
            pred_batch = model.predict(batch)
            predictions.extend(pred_batch)
            
            if i % 20000 == 0:
                logger.info(f"Processed {i}/{len(X_test)} records...")
        
        predictions = np.array(predictions)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_data['id'],
            'PremiumAmount': predictions
        })
        
        # Save submission
        submission_file = 'output/svr_only_submission.csv'
        submission.to_csv(submission_file, index=False)
        
        logger.info(f"\nSubmission saved to: {submission_file}")
        logger.info(f"Predictions range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        logger.info(f"Mean prediction: ${predictions.mean():.2f}")
        logger.info(f"Median prediction: ${np.median(predictions):.2f}")
        logger.info(f"Std deviation: ${predictions.std():.2f}")
        
        # Check for extreme values
        outliers = predictions[predictions > predictions.mean() + 3 * predictions.std()]
        if len(outliers) > 0:
            logger.warning(f"Found {len(outliers)} predictions > ${predictions.mean() + 3 * predictions.std():.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating SVR submission: {str(e)}")
        raise

if __name__ == "__main__":
    generate_svr_submission()
