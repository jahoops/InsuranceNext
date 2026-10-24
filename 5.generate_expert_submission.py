import pandas as pd
import numpy as np
import json
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_local_experts(train_data: pd.DataFrame, test_record: pd.DataFrame, 
                      k_neighbors: int = 50) -> tuple:
    """Find training records most similar to test record"""
    scaler = RobustScaler(quantile_range=(1, 99))
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_record.reshape(1, -1))
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean', n_jobs=1)
    nbrs.fit(scaled_train)
    distances, indices = nbrs.kneighbors(scaled_test)
    
    return indices[0], distances[0]

def evaluate_model_locally(model, X_local, y_local) -> float:
    """Evaluate model's performance on local neighborhood"""
    try:
        pred = model.predict(X_local)
        errors = np.abs(pred - y_local) / (y_local + 1)
        return np.mean(errors)
    except:
        return float('inf')

def process_chunk(args):
    """Process a chunk of test data"""
    chunk_data, train_data, train_target, models_info, chunk_id = args
    
    results = []
    for idx, test_record in enumerate(chunk_data.values):
        # Find similar training records
        similar_indices, distances = find_local_experts(train_data, test_record)
        X_local = train_data.iloc[similar_indices]
        y_local = train_target.iloc[similar_indices]
        
        # Evaluate each model on local neighborhood
        model_scores = {}
        for name, info in models_info.items():
            model = info['model']
            local_error = evaluate_model_locally(model, X_local, y_local)
            model_scores[name] = local_error
        
        # Select best model for this record
        best_model_name = min(model_scores.items(), key=lambda x: x[1])[0]
        best_model = models_info[best_model_name]['model']
        
        # Get prediction
        prediction = best_model.predict(test_record.reshape(1, -1))[0]
        results.append({
            'prediction': prediction,
            'best_model': best_model_name,
            'local_error': model_scores[best_model_name]
        })
        
        if idx % 100 == 0:
            logger.info(f"Chunk {chunk_id}: Processed {idx}/{len(chunk_data)} records")
    
    return results

def generate_expert_submission():
    try:
        # Load models configuration
        with open('output/best_models_config.json', 'r') as f:
            models_config = json.load(f)
        
        # Load data
        train_data = pd.read_csv('output/cleaned_train.csv')
        test_data = pd.read_csv('output/cleaned_test.csv')
        
        X_train = train_data.drop('PremiumAmount', axis=1)
        y_train = train_data['PremiumAmount']
        
        # Load all models
        models_info = {}
        for name, config in models_config.items():
            try:
                model_info = joblib.load(config['path'])
                if model_info['status'] == 'success':
                    models_info[name] = {
                        'model': model_info['model'],
                        'score': config['score']
                    }
            except Exception as e:
                logger.warning(f"Couldn't load {name}: {e}")
        
        logger.info(f"Loaded {len(models_info)} models successfully")
        
        # Process test data in parallel chunks
        chunk_size = 1000
        n_chunks = len(test_data) // chunk_size + 1
        chunks = np.array_split(test_data, n_chunks)
        
        # Prepare arguments for parallel processing
        chunk_args = [
            (chunk, X_train, y_train, models_info, i) 
            for i, chunk in enumerate(chunks)
        ]
        
        # Process chunks in parallel
        all_results = []
        with ProcessPoolExecutor() as executor:
            for chunk_results in executor.map(process_chunk, chunk_args):
                all_results.extend(chunk_results)
        
        # Combine results
        predictions = [r['prediction'] for r in all_results]
        used_models = [r['best_model'] for r in all_results]
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_data['id'],
            'PremiumAmount': predictions
        })
        
        # Log statistics
        logger.info("\nModel Usage Statistics:")
        model_usage = pd.Series(used_models).value_counts()
        for model, count in model_usage.items():
            percentage = count / len(test_data) * 100
            logger.info(f"{model}: {count} records ({percentage:.1f}%)")
        
        # Save submission
        output_path = Path('output/expert_ensemble_submission.csv')
        submission.to_csv(output_path, index=False)
        logger.info(f"\nSaved submission to {output_path}")
        
        # Save detailed results
        details = pd.DataFrame(all_results)
        details['id'] = test_data['id']
        details.to_csv('output/expert_ensemble_details.csv', index=False)
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating expert submission: {str(e)}")
        return False

if __name__ == "__main__":
    generate_expert_submission()
