"""
Usage:
    # Analyze all version folders
    python check_models.py
    
    # Analyze specific version
    python check_models.py --versions v1
    
    # Compare multiple specific versions
    python check_models.py --versions v1 v2 v3
    
    # Compare latest two versions
    python check_models.py --versions $(ls -d output/v* | tail -n 2 | xargs basename)

Output:
    - Prints analysis to console
    - Saves cross_version_analysis.json to latest version folder
    - Shows:
        - Top 5 models across all versions
        - Best model per version
        - Most commonly used features
        - Error analysis if any models failed
"""

import joblib
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm
import subprocess
import json  # Add this import
import tensorflow as tf  # Add this import if not already present

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_all_model_files(base_dir: Path) -> list:
    """Recursively find all model files in any subfolder"""
    model_files = []
    for path in base_dir.rglob('best_*_model.pkl'):
        try:
            # Test if file is loadable
            model_info = joblib.load(path)
            if isinstance(model_info, dict) and 'score' in model_info:
                model_files.append(path)
                logger.info(f"Found valid scikit-learn model: {path}")
        except Exception as e:
            logger.warning(f"Skipping invalid scikit-learn model file {path}: {e}")
    
    # Additionally, find Keras model files
    for path in base_dir.rglob('best_*_model.h5'):
        try:
            # Attempt to load the Keras model
            model_info = joblib.load(path.with_suffix('.pkl'))  # Assuming a corresponding .pkl exists
            if 'score' in model_info:
                model_files.append(path)
                logger.info(f"Found valid Keras model: {path}")
        except Exception as e:
            logger.warning(f"Skipping invalid Keras model file {path}: {e}")
    
    return model_files

def extract_model_info(model_path: Path) -> dict:
    """Extract relevant information from model files"""
    try:
        if model_path.suffix == '.pkl':
            model_info = joblib.load(model_path)
            if 'score' not in model_info:
                raise KeyError("'score' key is missing in model_info")
        elif model_path.suffix == '.h5':
            # Load corresponding .pkl for Keras models
            pkl_path = model_path.with_suffix('.pkl')
            if pkl_path.exists():
                model_info = joblib.load(pkl_path)
            else:
                raise FileNotFoundError(f"Corresponding .pkl file for {model_path} not found.")
        else:
            logger.warning(f"Unsupported file type: {model_path}")
            return {}
        
        return {
            'Version': model_info.get('version', 'unknown'),
            'Model': model_info.get('name', 'unknown'),
            'RMSLE Score': model_info.get('score', float('inf')),
            'Feature Count': len(model_info.get('features', [])),
            'Status': model_info.get('status', 'unknown'),
            'Error': model_info.get('error', None),
            'Path': str(model_path),
            # ...add other relevant fields if necessary...
        }
    except Exception as e:
        logger.error(f"Error extracting info from {model_path}: {e}")
        return {}

def generate_cross_version_analysis(base_dir: Path):
    """Generate cross_version_analysis.json with accurate scores"""
    model_files = find_all_model_files(base_dir)
    models_info = []
    feature_usage = {}
    
   
    # Update feature usage
    for info in models_info:
        features = info.get('features', [])
        for feature in features:
            if feature == 'id':
                continue  # Exclude 'id' from feature usage
            feature_usage[feature] = feature_usage.get(feature, 0) + 1
    
    # Determine best overall model
    successful_models = [m for m in models_info if m['Status'] == 'success']
    if successful_models:
        best_overall = min(successful_models, key=lambda x: x['RMSLE Score'])
    else:
        best_overall = {}
    
    # Determine best model by type
    best_by_type = {}
    for model in models_info:
        model_name = model['Model']
        if model_name not in best_by_type or model['RMSLE Score'] < best_by_type[model_name]['Best Score']:
            best_by_type[model_name] = {
                'Model': model_name,
                'Best Score': model['RMSLE Score'],
                'Version': model['Version'],
                'Sample Size': model.get('Sample Size', None),
                'Status': model['Status']
            }
    
    # Prepare ensemble recommendation
    ensemble_recommendation = []
    total_weight = 0
    for model_name, details in best_by_type.items():
        if details['Status'] == 'success' and details['Best Score'] < float('inf'):
            weight = 1 / details['Best Score'] if details['Best Score'] != 0 else 0.0
            ensemble_recommendation.append({
                'Model': model_name,
                'Best Score': details['Best Score'],
                'Version': details['Version'],
                'Sample Size': details['Sample Size'],
                'Status': details['Status'],
                'Weight': weight
            })
            total_weight += weight
        else:
            ensemble_recommendation.append({
                'Model': model_name,
                'Best Score': details['Best Score'],
                'Version': details['Version'],
                'Sample Size': details['Sample Size'],
                'Status': details['Status'],
                'Weight': 0.0
            })
    
    # Normalize weights
    for model in ensemble_recommendation:
        model['Weight'] = model['Weight'] / total_weight if total_weight > 0 else 0.0
    
    # Calculate expected ensemble score (weighted average)
    expected_score = sum([m['Best Score'] * m['Weight'] for m in ensemble_recommendation])
    
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'versions_analyzed': list(set([m['Version'] for m in models_info])),
        'models': models_info,
        'feature_usage': feature_usage,
        'best_overall_model': best_overall,
        'best_by_type': list(best_by_type.values()),
        'ensemble_recommendation': {
            'models': ensemble_recommendation,
            'total_models': len(ensemble_recommendation),
            'expected_score': expected_score
        }
    }
    
    # Save to JSON
    with open(base_dir / 'cross_version_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info("cross_version_analysis.json generated successfully.")

def run_check_models():
    base_dir = Path('output')  # Adjust if necessary
    generate_cross_version_analysis(base_dir)

if __name__ == "__main__":
    run_check_models()

import os
import json
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_version(path):
    """Extract version from the output directory path."""
    return os.path.basename(os.path.dirname(path))

def extract_model_name(model_info):
    """Extract model name from model_info."""
    return model_info.get('name', 'unknown')

def main():
    analysis = {
        "timestamp": datetime.utcnow().isoformat(),
        "versions_analyzed": [],
        "models": [],
        "feature_usage": {},
        "best_overall_model": {},
        "best_by_type": [],
        "ensemble_recommendation": {}
    }
    
    models = []
    best_overall = {"score": float('inf')}
    best_by_type = {}
    
    # Traverse the output directory to find all model_info files
    for root, dirs, files in os.walk('output'):
        for file in files:
            if file.endswith('_model.pkl'):
                model_path = os.path.join(root, file)
                try:
                    model_info = joblib.load(model_path)
                    
                    version = extract_version(model_path)
                    model_name = extract_model_name(model_info)
                    rmsle = model_info.get('score', float('inf'))
                    feature_count = len(model_info.get('features', []))
                    status = model_info.get('status', 'unknown')
                    error = model_info.get('error', None)
                    
                    models.append({
                        "Version": version,
                        "Model": model_name,
                        "RMSLE Score": rmsle,
                        "Feature Count": feature_count,
                        "Status": status,
                        "Error": error,
                        "Path": model_path
                    })
                    
                    # Update best overall model
                    if rmsle < best_overall["score"]:
                        best_overall = {
                            "Version": version,
                            "Model": model_name,
                            "RMSLE Score": rmsle,
                            "Feature Count": feature_count,
                            "Status": status,
                            "Error": error,
                            "Path": model_path
                        }
                    
                    # Update best by type
                    if model_name not in best_by_type or rmsle < best_by_type[model_name]["Best Score"]:
                        best_by_type[model_name] = {
                            "Model": model_name,
                            "Best Score": rmsle,
                            "Version": version,
                            "Sample Size": model_info.get('sample_size', None),
                            "Status": status
                        }
                    
                except Exception as e:
                    logger.error(f"Failed to load {model_path}: {str(e)}")
    
    # Populate analysis
    analysis["versions_analyzed"] = list(set([model["Version"] for model in models]))
    analysis["models"] = models
    analysis["best_overall_model"] = best_overall
    analysis["best_by_type"] = list(best_by_type.values())
    
    # Ensemble Recommendation (Example: Using the best overall model)
    analysis["ensemble_recommendation"] = {
        "models": [
            {
                "Model": best_overall["Model"],
                "Best Score": best_overall["RMSLE Score"],
                "Version": best_overall["Version"],
                "Sample Size": best_overall.get("Sample Size", None),
                "Status": best_overall["Status"],
                "Weight": 1.0
            }
        ],
        "total_models": 1,
        "expected_score": best_overall["RMSLE Score"]
    }
    
    # Save to JSON
    with open('output/cross_version_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info("cross_version_analysis.json has been updated successfully.")

if __name__ == "__main__":
    from datetime import datetime
    main()