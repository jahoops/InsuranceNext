import pandas as pd

def test_feature_columns_consistency():
    with open('output/feature_columns.txt', 'r') as f:
        feature_columns = f.read().splitlines()
    
    train = pd.read_csv('output/cleaned_train.csv')
    test = pd.read_csv('output/cleaned_test.csv')
    
    assert all(col in train.columns for col in feature_columns), "Not all feature columns are present in cleaned_train.csv."
    assert all(col in test.columns for col in feature_columns), "Not all feature columns are present in cleaned_test.csv."