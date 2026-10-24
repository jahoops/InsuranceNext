import joblib
import numpy as np
import pandas as pd
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_weighted_ensemble():
    # Load test data
    test = pd.read_csv('output/cleaned_test.csv')
    
    # Define models and their weights (inverse of RMSLE)
    models = {
        'RandomForest': 1/1.1424662354316557,  # Updated weight
        'SVR': 1/1.085446930530816,
        'NeuralNet': 1/1.085446930530816  # Example if applicable
    }
    
    # Normalize weights
    total_weight = sum(models.values())
    weights = {k: v/total_weight for k, v in models.items()}
    
    predictions = []
    for model_name, weight in weights.items():
        try:
            model_path = f'output/best_{model_name}_model_v1.pkl'
            model_info = joblib.load(model_path)
            model = model_info['model']
            features = model_info['features']
            
            # Ensure test data has correct features
            X_test = test[features]
            
            # Make prediction
            pred = model.predict(X_test)
            pred = np.maximum(pred, 0)  # Ensure non-negative
            predictions.append(pred * weight)
            logger.info(f"{model_name} prediction range: {pred.min():.2f} - {pred.max():.2f}")
            
        except Exception as e:
            logger.error(f"Error with {model_name}: {str(e)}")
            continue
    
    final_predictions = np.sum(predictions, axis=0)
    
    submission = pd.DataFrame({
        'id': test['id'],
        'PremiumAmount': final_predictions
    })
    
    submission.to_csv('output/weighted_ensemble_submission.csv', index=False)
    logger.info(f"Created weighted ensemble submission with prediction range: {final_predictions.min():.2f} - {final_predictions.max():.2f}")

def run_others():
    # ...existing code for multiple smaller models...
    
    # Train each model
    # ...existing code...
    
    # Save model/results
    # ...existing code...
    
    return "Others completed successfully"

if __name__ == "__main__":
    create_weighted_ensemble()
    run_others()