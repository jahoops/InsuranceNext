from abc import ABC, abstractmethod
from pathlib import Path
from .utils import safe_rmsle, save_model, load_data as utils_load_data, log_memory_usage  # Ensure log_memory_usage is imported
import logging
import traceback
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseExperiment(ABC):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_data(self):
        """Load data with logging"""
        self.logger.info(f"Loading data for {self.name}")
        return utils_load_data()
        
    def log_memory_usage(self, stage: str):
        """Wrapper for the log_memory_usage function."""
        log_memory_usage(stage)
        
    def run(self) -> dict:
        """Template method pattern with enhanced error handling and memory logging"""
        try:
            self.log_memory_usage(f"{self.name} - Before loading data")
            X, y = self.load_data()
            self.log_memory_usage(f"{self.name} - After loading data")
            
            self.logger.info(f"Creating model for {self.name}")
            self.log_memory_usage(f"{self.name} - Before creating model")
            model = self.create_model()
            self.log_memory_usage(f"{self.name} - After creating model")
            
            self.logger.info(f"Training {self.name}")
            self.log_memory_usage(f"{self.name} - Before training model")
            model.fit(X, y)
            self.log_memory_usage(f"{self.name} - After training model")
            
            self.logger.info(f"Evaluating {self.name}")
            predictions = model.predict(X)
            score = safe_rmsle(y, predictions)
            self.log_memory_usage(f"{self.name} - After evaluation")
            self.logger.info(f"{self.name} score: {score:.4f}")
            
            model_info = self.create_model_info(model, X, y, score)
            
            try:
                save_model(model_info, self.name, self.output_dir)
                self.log_memory_usage(f"{self.name} - After saving model")
            except Exception as e:
                self.logger.error(f"Failed to save {self.name} model: {str(e)}")
                model_info['status'] = 'success_but_not_saved'
                model_info['error'] = str(e)
                self.log_memory_usage(f"{self.name} - After failed model save")
            
            return model_info
            
        except Exception as e:
            error_msg = f"{self.name} failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            self.log_memory_usage(f"{self.name} - On failure")
            return {
                'model': None,
                'features': [],
                'score': float('inf'),
                'status': 'failed',
                'error': error_msg
            }
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the experiment"""
        pass
        
    @abstractmethod
    def create_model(self):
        """Create and configure the model"""
        pass

    def create_model_info(self, model, X, y, score, additional_info=None):
        """Create standardized model info dictionary"""
        info = {
            'name': self.name,
            'model': model,
            'features': X.columns.tolist(),
            'score': score,
            'status': 'success',
            'error': None,
            'version': str(self.output_dir),
            'sample_size': len(X),
            'feature_count': len(X.columns),
            'timestamp': datetime.now().isoformat(),
            'parameters': getattr(model, 'get_params', lambda: {})()
        }
        
        if additional_info:
            info.update(additional_info)
            
        return info

    def save_state(output_dir, completed_experiments, failed_experiments, crashed_experiments):
        """Save experiment state.
        
        Args:
            completed_experiments: list of dicts with 'name' and 'score'
            failed_experiments: list of tuples (name, error)
            crashed_experiments: list of experiment names (strings)
        """
        state_data = {
            "completed": [{"name": exp['name'], "score": exp['score']} for exp in completed_experiments],  # Ensure 'score' is included
            "failed": failed_experiments,
            "crashed": list(crashed_experiments),
            "timestamp": datetime.now().isoformat()
        }
        
        # ...existing code...
        
        with open(output_dir / "experiment_state.json", "w") as f:
            json.dump(state_data, f, indent=2)
        
        # ...existing code...

feature_importances = [
    ("HealthScore", 0.5523),
    ("AnnualIncome", 0.2245),
    ("CreditScore", 0.0800),
    ("Age", 0.0085),
    ("SignupYear", 0.0023),
    ("PreviousClaims", 0.0021),
    ("NumberofDependents", 0.0018),
    ("Location_Rural", 0.0015),
    ("VehicleAge", 0.0015),
    ("InsuranceDuration", 0.0015),
    ("CustomerFeedback_Average", 0.0014),
    ("SmokingStatus_No", 0.0013),
    ("PropertyType_House", 0.0013),
    ("SignupMonth", 0.0012),
    ("Occupation_SelfEmployed", 0.0011),
    ("MaritalStatus_Married", 0.0011),
    ("Occupation_nan", 0.0009),
    ("CustomerFeedback_Good", 0.0007),
    ("EducationLevel_PhD", 0.0006),
    ("PropertyType_Apartment", 0.0005),
    ("Location_Urban", 0.0005),
    ("ExerciseFrequency_Daily", 0.0004)
]