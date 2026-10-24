import os
import json
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import logging
import joblib  # Add this import

# Custom RMSLE metric
def rmsle(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))))

try:
    # Load and prepare data
    data = pd.read_csv('output/cleaned_train.csv')
    y = data['PremiumAmount']
    X = data.drop(columns=['PremiumAmount', 'id'])

    # Load selected features
    with open('selected_features.txt', 'r') as f:
        selected_features = f.read().splitlines()  # Add this block

    X = X[selected_features]  # Modify this line to select features

    # Scale features using the scaler fitted on training data
    robust_scaler = joblib.load('output/robust_scaler.pkl')  # Load scaler from 'output/'
    X_scaled = robust_scaler.transform(X)  # Ensure feature order matches
    
    # Split data (optional, if you want to have a validation set)
    # Here, we'll train on the entire dataset
    X_train_scaled = X_scaled
    y_train = y

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Look in multiple possible locations for model files
    def find_model_files(trial_dir):
        """Search for model files in various possible locations"""
        possible_weight_files = [
            os.path.join(trial_dir, 'checkpoint.weights.h5'),
            os.path.join(trial_dir, 'checkpoint'),
            os.path.join(trial_dir, 'best_model.h5'),
            os.path.join(trial_dir, 'best_model.keras'),
        ]

        # Return the first file that exists
        for file in possible_weight_files:
            if os.path.exists(file):
                return file
        return None

    # Find all trial directories and sort them numerically
    base_dir = '/home/USER/Kaggle/InsuranceNext/my_dir/insurance_tuning'
    checkpoint_dir = '/home/USER/Kaggle/InsuranceNext/checkpoints'
    trial_dirs = sorted(glob.glob(f"{base_dir}/trial_*"), 
                       key=lambda x: int(x.split('_')[-1]))

    # Track best performance and all trials
    best_val_rmsle = float('inf')
    best_trial_info = None
    all_trials = []

    print("\nExamining all trials:")
    print("-" * 50)

    # First check trials directory
    for trial_dir in trial_dirs:
        trial_file = os.path.join(trial_dir, 'trial.json')
        if os.path.exists(trial_file):
            with open(trial_file, 'r') as f:
                trial_data = json.load(f)

                trial_num = os.path.basename(trial_dir).split('_')[-1]
                val_rmsle = trial_data.get('score', None)
                status = trial_data.get('status', 'UNKNOWN')
                weights_file = find_model_files(trial_dir)

                if val_rmsle is not None and weights_file is not None:
                    trial_info = {
                        'trial': trial_num,
                        'rmsle': val_rmsle,
                        'status': status,
                        'weights': weights_file,
                        'dir': trial_dir,
                        'trial_json': trial_file
                    }
                    all_trials.append(trial_info)

                    print(f"Trial {trial_num:>4}: RMSLE = {val_rmsle:.6f} ({status})")

                    if val_rmsle < best_val_rmsle:
                        best_val_rmsle = val_rmsle
                        best_trial_info = trial_info
                        print(f"   ↳ New best trial!")

    # Also check checkpoints directory for any saved models
    if os.path.exists(checkpoint_dir):
        for checkpoint_path in glob.glob(f"{checkpoint_dir}/**/best_model*.h5", recursive=True):
            try:
                model = load_model(checkpoint_path, custom_objects={'rmsle': rmsle})
                # Evaluate model if needed
                logger.info(f"Found checkpoint model: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint model {checkpoint_path}: {e}")

    # Sort and display all trials by RMSLE
    print("\nTop 10 trials by RMSLE:")
    print("-" * 50)
    sorted_trials = sorted(all_trials, key=lambda x: x['rmsle'])
    for trial in sorted_trials[:10]:
        print(f"Trial {trial['trial']:>4}: RMSLE = {trial['rmsle']:.6f} ({trial['status']})")
        print(f"        Weights: {trial['weights']}")

    if best_trial_info:
        print(f"\nBest trial found: {best_trial_info['dir']}")
        print(f"Best validation RMSLE: {best_val_rmsle}")
        print(f"Loading model from: {best_trial_info['weights']}")

        # Load configuration from the best trial
        with open(best_trial_info['trial_json'], 'r') as f:
            trial_data = json.load(f)
            hp_values = trial_data['hyperparameters']['values']

        # Recreate the model architecture using inferred input_shape
        model = Sequential([
            Dense(
                units=hp_values['units_1'], 
                activation='relu', 
                kernel_regularizer=tf.keras.regularizers.l2(hp_values['l2_1']),
                input_shape=(X.shape[1],)
            ),
            BatchNormalization(),
            Dropout(hp_values['dropout_1']),

            Dense(
                units=hp_values['units_2'], 
                activation='relu', 
                kernel_regularizer=tf.keras.regularizers.l2(hp_values['l2_2'])
            ),
            BatchNormalization(),
            Dropout(hp_values['dropout_2']),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='relu')
        ])

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=hp_values['learning_rate']),
            loss=rmsle,
            metrics=['mean_absolute_error', rmsle]
        )

        # Load the weights from the best trial
        try:
            model.load_weights(best_trial_info['weights'])
            model.save('best_insurance_model.keras')
            print("\nBest model successfully saved as best_insurance_model.keras")
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            raise
    else:
        print("No valid trials found!")

except Exception as e:
    logger.error(f"Error in find_best.py: {str(e)}")
    print("No valid trials found!")
    raise

try:
    # Load selected features
    with open('selected_features.txt', 'r') as f:
        selected_features = f.read().splitlines()  # Add this block

    X = X[selected_features]  # Modify this line to select features

    # Scale features using the scaler fitted on training data
    robust_scaler = joblib.load('output/robust_scaler.pkl')  # Load scaler from 'output/'
    X_scaled = robust_scaler.transform(X)  # Ensure feature order matches
    
except Exception as e:
    logger.error(f"Error in find_best.py: {str(e)}")
    raise