import os

def test_input_files_exist():
    input_dir = 'input'
    required_files = ['train.csv', 'test.csv']
    for file in required_files:
        assert os.path.exists(os.path.join(input_dir, file)), f"Input file {file} is missing."