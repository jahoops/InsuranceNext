import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_model(model_path='output/v1/best_XGBoost_model.pkl'):
    try:
        model_info = joblib.load(model_path)
        
        print("\n=== XGBoost Model Info ===")
        print(f"Keys in model_info: {list(model_info.keys())}")
        print(f"Model Name: {model_info.get('name')}")
        print(f"Score: {model_info.get('score')}")
        print(f"Features Used: {model_info.get('features')}")
        print(f"\nModel score (RMSLE): {model_info.get('score', 'Not found')}")
        print(f"Model status: {model_info.get('status', 'Not found')}")
        print(f"Sample size used: {model_info.get('sample_size', 'Not found')}")
        print(f"\nPipeline steps: {[step[0] for step in model_info['model'].named_steps.items()]}")
        print(f"\nNumber of features: {len(model_info.get('features', []))}")
        print("\nFeatures used:")
        for feat in model_info.get('features', []):
            print(f"- {feat}")
            
    except Exception as e:
        logger.error(f"Error inspecting model: {str(e)}")

if __name__ == "__main__":
    inspect_model()
