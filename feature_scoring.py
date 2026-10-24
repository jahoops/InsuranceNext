
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def score_features(data_path):
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Separate target from features
    # Assuming "PremiumAmount" is the target
    y = df['PremiumAmount']
    X = df.drop(columns=['PremiumAmount', 'id'], errors='ignore')
    
    # Calculate mutual information scores
    scores = mutual_info_regression(X, y, discrete_features=False)
    
    # Match each feature to its score
    feature_scores = sorted(zip(X.columns, scores), key=lambda x: x[1], reverse=True)
    
    # Print results
    for feature, score in feature_scores:
        print(f"{feature}: {score:.4f}")

if __name__ == "__main__":
    score_features("/home/USER/Kaggle/InsuranceNext/output/cleaned_train.csv")