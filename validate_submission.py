import pandas as pd
import numpy as np
import logging
import joblib  # Ensure this import is present

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_rmsle(y_true, y_pred):
    """Calculate RMSLE the same way Kaggle does"""
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

try:
    # Load original training data to get actual premium distribution
    train_data = pd.read_csv('input/train.csv')  # Changed from 'data/train.csv' to 'input/train.csv'
    train_premiums = train_data['PremiumAmount']
    
    # Load selected features
    with open('output/feature_columns.txt', 'r') as f:  # Changed from 'selected_features.txt' to 'output/feature_columns.txt'
        selected_features = f.read().splitlines()  # Add this block

    # Load your submission
    submission = pd.read_csv('submission.csv')

    # Ensure submission has the correct columns
    expected_columns = ['id', 'PremiumAmount']
    if not all(col in submission.columns for col in expected_columns):
        raise ValueError(f"Submission file must contain columns: {expected_columns}")

    # Basic validation checks
    logger.info("\nSubmission Statistics vs Training Data:")
    logger.info(f"{'Metric':<15} {'Submission':<15} {'Training':<15}")
    logger.info("-" * 45)
    logger.info(f"{'Min:':<15} {submission['PremiumAmount'].min():>15.2f} {train_premiums.min():>15.2f}")
    logger.info(f"{'Max:':<15} {submission['PremiumAmount'].max():>15.2f} {train_premiums.max():>15.2f}")
    logger.info(f"{'Mean:':<15} {submission['PremiumAmount'].mean():>15.2f} {train_premiums.mean():>15.2f}")
    logger.info(f"{'Median:':<15} {submission['PremiumAmount'].median():>15.2f} {train_premiums.median():>15.2f}")
    logger.info(f"{'Std:':<15} {submission['PremiumAmount'].std():>15.2f} {train_premiums.std():>15.2f}")
    
    # Check for invalid values
    n_zeros = (submission['PremiumAmount'] == 0).sum()
    n_negatives = (submission['PremiumAmount'] < 0).sum()
    n_missing = submission['PremiumAmount'].isnull().sum()
    n_infinite = np.isinf(submission['PremiumAmount']).sum()
    
    logger.info("\nPotential Issues:")
    if n_zeros > 0: logger.warning(f"Found {n_zeros} zero premiums")
    if n_negatives > 0: logger.warning(f"Found {n_negatives} negative premiums")
    if n_missing > 0: logger.warning(f"Found {n_missing} missing values")
    if n_infinite > 0: logger.warning(f"Found {n_infinite} infinite values")
    
    # Distribution comparison
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    logger.info("\nPercentile Comparison:")
    logger.info(f"{'Percentile':<10} {'Submission':<15} {'Training':<15}")
    logger.info("-" * 40)
    for p in percentiles:
        sub_val = np.percentile(submission['PremiumAmount'], p)
        train_val = np.percentile(train_premiums, p)
        logger.info(f"{p:>3}%{'':<7} {sub_val:>15.2f} {train_val:>15.2f}")

except Exception as e:
    logger.error(f"Error validating submission: {str(e)}")
    raise
