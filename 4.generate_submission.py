import pandas as pd
import numpy as np
import json
import joblib
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

def get_confidence_score(predictions: np.ndarray, models_info: dict) -> np.ndarray:
    """Calculate confidence score for each prediction based on model characteristics"""
    confidences = []
    
    # Get median prediction for each row
    median_preds = np.median(predictions, axis=0)
    
    for i, model_preds in enumerate(predictions):
        # Calculate factors that influence confidence:
        # 1. How close to median (closer = more confident)
        deviation_from_median = np.abs(model_preds - median_preds) / (median_preds + 1e-8)
        # 2. Model's overall score (better score = more confident)
        model_score = list(models_info.values())[i]['score']
        # 3. Prediction volatility (less volatile = more confident)
        volatility = np.std(model_preds) / (np.mean(model_preds) + 1e-8)
        
        # Combine factors into confidence score
        confidence = (1 / (model_score + 1e-8)) * (1 / (deviation_from_median + 1e-8)) * (1 / (volatility + 1e-8))
        confidences.append(confidence)
    
    return np.array(confidences)

def get_scaling_factors(predictions: np.ndarray, model_names: list) -> np.ndarray:
    """Calculate scaling factors to normalize predictions to similar ranges"""
    medians = np.median(predictions, axis=1)
    # Use SVR's median as the target since it has best real-world performance
    svr_idx = model_names.index('SVR')
    target_median = medians[svr_idx]
    
    # Calculate scaling factors
    scaling_factors = target_median / (medians + 1e-8)
    return scaling_factors[:, np.newaxis]  # Broadcasting shape

def generate_submission():
    try:           
        test_data = pd.read_csv('output/cleaned_test.csv')
        all_predictions = []
        model_names = []
        original_scores = []
        # Load model performance report
        model_performance_path = 'output/v1/model_performance_report.json'
        with open(model_performance_path, 'r') as f:
            model_performance = json.load(f)

        # Extract model names and configurations
        models_config = {
            model['name']: {
                'path': model['path'],
                'score': model['score']
            }
            for model in model_performance['models']
        }
        # Generate predictions from each model
        for model_name, config in models_config.items():
            try:
                logger.info(f"\nProcessing {model_name}...")
                model_path = config['path']
                model_info = joblib.load(model_path)
                model = model_info['model']
                features = model_info['features']
                score = config['score']
                
                # Get predictions
                X_test = test_data[features]
                pred = model.predict(X_test)
                pred = np.maximum(pred, 0)
                
                all_predictions.append(pred)
                model_names.append(model_name)
                original_scores.append(score)
                
                logger.info(f"{model_name} prediction stats:")
                logger.info(f"  Range: {pred.min():.2f} - {pred.max():.2f}")
                logger.info(f"  Mean: {pred.mean():.2f}")
                logger.info(f"  Median: {np.median(pred):.2f}")
                
            except Exception as e:
                logger.error(f"Error using {model_name}: {str(e)}")
                continue
        
        if not all_predictions:
            raise ValueError("No valid predictions generated")
        
        # Convert to array and scale predictions
        predictions = np.array(all_predictions)
        scaling_factors = get_scaling_factors(predictions, model_names)
        scaled_predictions = predictions * scaling_factors
        
        # Recalculate weights based on scores
        scores = np.array(original_scores)
        weights = 1 / (scores + 1e-8)
        weights = weights / np.sum(weights)
        
        # Apply weights with extra weight to SVR
        svr_idx = model_names.index('SVR')
        weights[svr_idx] *= 2  # Double SVR's weight
        weights = weights / np.sum(weights)  # Renormalize
        
        # Generate weighted predictions
        weighted_preds = scaled_predictions * weights[:, np.newaxis]
        final_predictions = np.sum(weighted_preds, axis=0)
        
        # Log contribution analysis
        logger.info("\nModel Contributions:")
        for i, name in enumerate(model_names):
            contribution = weighted_preds[i].mean()
            logger.info(f"{name}:")
            logger.info(f"  Weight: {weights[i]:.4f}")
            logger.info(f"  Avg Contribution: {contribution:.2f}")
            logger.info(f"  Scaling Factor: {scaling_factors[i][0]:.4f}")
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_data['id'],
            'PremiumAmount': final_predictions
        })
        
        # Save submission with version info
        output_path = Path('output') / f'balanced_ensemble_submission.csv'
        submission.to_csv(output_path, index=False)
        
        logger.info(f"\nSubmission saved to {output_path}")
        logger.info(f"Final predictions:")
        logger.info(f"  Range: {final_predictions.min():.2f} - {final_predictions.max():.2f}")
        logger.info(f"  Mean: {final_predictions.mean():.2f}")
        logger.info(f"  Median: {np.median(final_predictions):.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating submission: {str(e)}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_submission()
