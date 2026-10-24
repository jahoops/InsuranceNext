import os
import json
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
import logging
import sys  # Add this import for graceful exit
import joblib  # Add this import

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rmsle(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))))

def evaluate_model(model, X_val, y_val):
    """Evaluate a model on validation data"""
    try:
        # Generate predictions
        y_pred_log = model.predict(X_val, verbose=0)
        y_pred = np.expm1(y_pred_log.flatten())
        
        # Calculate RMSLE
        val_rmsle = np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_val))))
        return val_rmsle, y_pred
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return float('inf'), None

def build_model_from_hyperparameters(hp_values, input_shape):
    """Recreate the model architecture based on hyperparameters"""
    try:
        model = Sequential([
            Dense(
                units=hp_values['units_1'],
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(hp_values['l2_1']),
                input_shape=(input_shape,)
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

        optimizer = Adam(learning_rate=hp_values['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss=rmsle,
            metrics=[rmsle]
        )
        return model
    except Exception as e:
        logger.error(f"Error building model from hyperparameters: {e}")
        raise

try:
    # Load preprocessing objects
    robust_scaler = joblib.load('output/robust_scaler.pkl')  # Load the robust scaler
    onehot_encoder = joblib.load('output/onehot_encoder.pkl')  # Load the one-hot encoder

    # Load feature columns used during training
    with open('output/feature_columns.txt', 'r') as f:
        feature_columns = f.read().splitlines()  # Load feature columns

    # Load selected numerical features
    with open('selected_features.txt', 'r') as f:
        selected_numeric_features = f.read().splitlines()  # Add this block

    # Load validation data
    validation_data = pd.read_csv('validation_set.csv')
    y_val = validation_data['PremiumAmount']  # Use original column name
    if 'id' in validation_data.columns:
        X_val = validation_data.drop(columns=['PremiumAmount', 'id'])
    else:
        X_val = validation_data.drop(columns=['PremiumAmount'])

    # Apply scaling to selected numeric features
    X_val_numeric = X_val[selected_numeric_features]  # Modify this line
    X_val_scaled = robust_scaler.transform(X_val_numeric)  # Scale selected numeric features
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val_numeric.columns)  # Create DataFrame

    # Remove one-hot encoding of categorical features
    # X_val_categorical = X_val.drop(columns=selected_numeric_features)
    # X_val_encoded = onehot_encoder.transform(X_val_categorical)  # Encode categorical features
    # X_val_encoded_df = pd.DataFrame(X_val_encoded, columns=onehot_encoder.get_feature_names_out())  # Create DataFrame

    # Combine scaled numeric features only
    X_val_processed = X_val_scaled_df  # Use only scaled numeric features

    # Reindex to match training feature columns, fill missing columns with 0
    # X_val = X_val_processed.reindex(columns=feature_columns, fill_value=0)  # Reindex to match features
    X_val = X_val_processed  # Directly use processed numeric features

    logger.info(f"Loaded and preprocessed validation data with shape: {X_val.shape}")
    
    # Infer input_shape from validation data
    input_shape = X_val.shape[1]
    logger.info(f"Inferred input_shape: {input_shape}")
    
    # Find all trial directories and sort them numerically
    base_dir = '/home/USER/Kaggle/InsuranceNext/tuner_directory/insurance_model_tuning'  # Corrected directory path
    trial_dirs = sorted(glob.glob(f"{base_dir}/trial_*"),
                        key=lambda x: int(x.split('_')[-1]))
    
    logger.info(f"Found {len(trial_dirs)} trial directories in '{base_dir}'.")

    # Track results
    trial_results = []

    print("\nEvaluating models on validation set:")
    print("-" * 50)

    # Evaluate each trial
    for trial_dir in trial_dirs:
        trial_file = os.path.join(trial_dir, 'trial.json')
        weights_file = os.path.join(trial_dir, 'checkpoint.weights.h5')
        
        trial_num = os.path.basename(trial_dir).split('_')[-1]
        logger.info(f"Processing Trial {trial_num}: Checking for required files.")
        
        # Check if necessary files exist
        if not os.path.exists(trial_file):
            logger.warning(f"Trial file not found for Trial {trial_num} at '{trial_file}'. Skipping.")
            continue
        if not os.path.exists(weights_file):
            logger.warning(f"Weights file not found for Trial {trial_num} at '{weights_file}'. Skipping.")
            continue
        
        try:
            # Load trial hyperparameters
            with open(trial_file, 'r') as f:
                trial_data = json.load(f)
            hp_values = trial_data['hyperparameters']['values']
            
            # Reconstruct the model architecture using inferred input_shape
            model = Sequential([
                Dense(
                    units=hp_values['units_1'],
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(hp_values['l2_1']),
                    input_shape=(input_shape,)
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

            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=hp_values['learning_rate']),
                loss=rmsle,
                metrics=[rmsle]
            )
            
            # Load model weights
            model.load_weights(weights_file)
            logger.info(f"Loaded weights for Trial {trial_num} from '{weights_file}'.")
            
            # Evaluate the model
            val_rmsle, predictions = evaluate_model(model, X_val, y_val)
            
            trial_results.append({
                'trial': trial_num,
                'rmsle': val_rmsle,
                'model_path': weights_file
            })
            
            print(f"Trial {trial_num:>4}: Validation RMSLE = {val_rmsle:.6f}")
            logger.info(f"Trial {trial_num}: Validation RMSLE = {val_rmsle:.6f}")
        
        except Exception as e:
            logger.error(f"Error processing Trial {trial_num}: {e}")

    # Also check checkpoint directory
    if os.path.exists(trial_dir):
        checkpoint_model = os.path.join(trial_dir, 'checkpoint.weights.h5')
        checkpoint_trial_file = os.path.join(trial_dir, 'trial.json')

        logger.info(f"Checking checkpoint model at '{checkpoint_model}'.")

        if os.path.exists(checkpoint_model) and os.path.exists(checkpoint_trial_file):
            try:
                # Load trial hyperparameters for checkpoint
                with open(checkpoint_trial_file, 'r') as f:
                    trial_data = json.load(f)
                hp_values = trial_data['hyperparameters']['values']

                # Reconstruct the model architecture using inferred input_shape
                model = Sequential([
                    Dense(
                        units=hp_values['units_1'],
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(hp_values['l2_1']),
                        input_shape=(input_shape,)
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

                # Compile the model
                model.compile(
                    optimizer=Adam(learning_rate=hp_values['learning_rate']),
                    loss=rmsle,
                    metrics=[rmsle]
                )

                # Load model weights
                model.load_weights(checkpoint_model)
                logger.info(f"Loaded weights for Checkpoint model from '{checkpoint_model}'.")

                # Evaluate the model
                val_rmsle, predictions = evaluate_model(model, X_val, y_val)
                
                trial_results.append({
                    'trial': 'checkpoint',
                    'rmsle': val_rmsle,
                    'model_path': checkpoint_model
                })
                
                print(f"Checkpoint model: Validation RMSLE = {val_rmsle:.6f}")
                logger.info(f"Checkpoint model: Validation RMSLE = {val_rmsle:.6f}")
            
            except Exception as e:
                logger.error(f"Error processing checkpoint model: {e}")
        else:
            logger.warning(f"Required checkpoint files not found in '{trial_dir}'.")
    else:
        logger.warning(f"Checkpoint directory '{trial_dir}' does not exist.")

    # Check if any trials were evaluated
    if not trial_results:
        logger.error("No models were evaluated. Please ensure that model weight files exist in the trial directories.")
        print("No models were found to evaluate. Please check the trial directories and model file names.")
        sys.exit(1)  # Exit gracefully

    # Sort and display results
    print("\nBest models by validation RMSLE:")
    print("-" * 50)
    sorted_results = sorted(trial_results, key=lambda x: x['rmsle'])
    
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"{i}. Trial {result['trial']:<4}: RMSLE = {result['rmsle']:.6f}")
        print(f"   Model: {result['model_path']}")
        logger.info(f"Rank {i}: Trial {result['trial']} with RMSLE = {result['rmsle']:.6f}")

    # Save best model
    best_result = sorted_results[0]
    try:
        # Reconstruct the best model architecture
        best_trial_dir = os.path.dirname(best_result['model_path'])
        best_trial_file = os.path.join(best_trial_dir, 'trial.json')
        
        with open(best_trial_file, 'r') as f:
            best_trial_data = json.load(f)
        hp_values = best_trial_data['hyperparameters']['values']
        
        # Recreate the model architecture using inferred input_shape
        best_model = Sequential([
            Dense(
                units=hp_values['units_1'],
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(hp_values['l2_1']),
                input_shape=(input_shape,)
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

        # Compile the model
        best_model.compile(
            optimizer=Adam(learning_rate=hp_values['learning_rate']),
            loss=rmsle,
            metrics=[rmsle]
        )

        # Load weights
        best_model.load_weights(best_result['model_path'])
        best_model.save('best_validated_model.keras')
        
        print(f"\nBest model saved as 'best_validated_model.keras'")
        print(f"Best validation RMSLE: {best_result['rmsle']:.6f}")
        print(f"From trial: {best_result['trial']}")
        logger.info(f"Best model from Trial {best_result['trial']} saved successfully.")
    except Exception as e:
        logger.error(f"Error saving best model: {e}")
        print(f"Error saving best model: {e}")
        raise

except Exception as e:
    logger.error(f"Error during validation: {str(e)}")
    raise
