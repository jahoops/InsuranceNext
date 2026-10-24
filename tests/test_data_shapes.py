import pandas as pd

def test_cleaned_train_shape():
    train = pd.read_csv('output/cleaned_train.csv')
    assert 'PremiumAmount' in train.columns, "PremiumAmount column is missing in cleaned_train.csv."
    assert train.shape[0] > 0, "cleaned_train.csv is empty."
    assert train.shape[1] > 2, "cleaned_train.csv does not have enough columns."

def test_cleaned_test_shape():
    test = pd.read_csv('output/cleaned_test.csv')
    assert 'id' in test.columns, "id column is missing in cleaned_test.csv."
    assert test.shape[0] > 0, "cleaned_test.csv is empty."
    assert test.shape[1] > 1, "cleaned_test.csv does not have enough columns."