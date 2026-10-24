import os
import pytest
import pandas as pd
import joblib

def test_cleaned_train_file():
    assert os.path.exists('output/cleaned_train.csv'), "cleaned_train.csv is missing."
    train_data = pd.read_csv('output/cleaned_train.csv')
    # ...existing code...
    assert 'PremiumAmount' in train_data.columns, "PremiumAmount column is missing."
    assert train_data.shape[0] > 0, "cleaned_train.csv is empty."

def test_cleaned_test_file():
    assert os.path.exists('output/cleaned_test.csv'), "cleaned_test.csv is missing."
    test_data = pd.read_csv('output/cleaned_test.csv')
    # ...existing code...
    assert 'id' in test_data.columns, "id column is missing."
    assert test_data.shape[0] > 0, "cleaned_test.csv is empty."

def test_robust_scaler_file():
    assert os.path.exists('output/robust_scaler.pkl'), "robust_scaler.pkl is missing."
    robust_scaler = joblib.load('output/robust_scaler.pkl')
    # ...existing code...
    assert hasattr(robust_scaler, 'transform'), "robust_scaler is not a valid scaler."

def test_onehot_encoder_file():
    assert os.path.exists('output/onehot_encoder.pkl'), "onehot_encoder.pkl is missing."
    onehot_encoder = joblib.load('output/onehot_encoder.pkl')
    # ...existing code...
    assert hasattr(onehot_encoder, 'transform'), "onehot_encoder is not a valid encoder."

def test_feature_columns_file():
    assert os.path.exists('output/feature_columns.txt'), "feature_columns.txt is missing."
    with open('output/feature_columns.txt', 'r') as f:
        feature_columns = f.read().splitlines()
    # ...existing code...
    assert len(feature_columns) > 0, "feature_columns.txt is empty."

def test_processed_data_consistency():
    train_data = pd.read_csv('output/cleaned_train.csv')
    test_data = pd.read_csv('output/cleaned_test.csv')
    # ...existing code...
    assert set(train_data.columns) == set(test_data.columns) | {'PremiumAmount'}, "Train and test data columns do not match."
