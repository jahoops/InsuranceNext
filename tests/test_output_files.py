import os

def test_output_directories_exist():
    output_dirs = ['output', 'logs']
    for directory in output_dirs:
        assert os.path.exists(directory), f"Output directory {directory} does not exist."

def test_output_files_generated():
    output_files = [
        'output/cleaned_train.csv',
        'output/cleaned_test.csv',
        'output/robust_scaler.pkl',
        'output/onehot_encoder.pkl',
        'output/feature_columns.txt'
    ]
    for file in output_files:
        assert os.path.exists(file), f"Output file {file} is missing."