import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
import tensorflow as tf
import keras_tuner as kt
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from keras_tuner import Objective  # Add this import
import os
from tensorflow.keras.models import load_model
from datetime import datetime
import json
import joblib  # Add this import
from tensorflow.keras.callbacks import Callback  # Corrected import
from experiments.utils import validate_data  # Import from utils instead of defining locally

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rmsle(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))))

try:
    # Load preprocessed training data
    data = pd.read_csv('output/cleaned_train.csv')  # Changed back to 'cleaned_train.csv'
    logger.info(f"Loaded training data with shape: {data.shape}")

    # Load selected features
    with open('output/feature_columns.txt', 'r') as f:  # Ensure the correct path
        selected_features = f.read().splitlines()

    # Separate features and target variable in training data
    X_train = data[selected_features]
    y_train = data['PremiumAmount']

    logger.info(f"Using {X_train.shape[1]} features for training")

    # Log transform target variables
    y_train_log = np.log1p(y_train)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train_log, test_size=0.2, random_state=42, shuffle=True
    )

    # Validate that X_train and y_train have the same number of samples
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Mismatch after split: X_train has {X_train.shape[0]} samples, "
                         f"but y_train has {y_train.shape[0]} samples.")

    # validate_data function checks the data integrity
    validate_data(X_train, y_train, stage="Training")

    # Infer input_shape from training data
    input_shape = X_train.shape[1]
    logger.info(f"Inferred input_shape: {input_shape}")

    # Define optimized model architecture
    def build_model(hp):
        inputs = Input(shape=(input_shape,))
        x = Dense(
            units=hp.Int('units_1', min_value=128, max_value=256, step=64),
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_1', 1e-5, 1e-2, sampling='log'))
        )(inputs)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float('dropout_1', 0.3, 0.6, step=0.1))(x)
        
        # Simplify model by reducing layers
        outputs = Dense(1, activation='relu')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        opt = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-3, sampling='log'))
        opt.build(model.trainable_variables)  # Build optimizer state

        model.compile(
            optimizer=opt,
            loss=rmsle,
            metrics=[rmsle]
        )
        return model

    # Define project name
    PROJECT_NAME = "insurance_model_tuning"
    
    # Define tuner directory
    TUNER_DIR = "tuner_directory"

    # At the beginning of the script, define a global variable to track the best validation score
    global best_val_rmsle  # Use the global variable
    best_val_rmsle = np.inf  # Initialize with infinity

    # Initialize tuner with your hypermodel
    tuner = kt.Hyperband(
        build_model,
        objective=Objective('val_rmsle', direction='min'),
        max_epochs=100,
        factor=3,
        hyperband_iterations=2,
        directory=TUNER_DIR,
        project_name=PROJECT_NAME,
        overwrite=True  # Set overwrite to True to reset tuner
    )

    # Add CSV logger for better tracking
    csv_logger = tf.keras.callbacks.CSVLogger('training_log.csv', append=True)

    # Define best model checkpoint callback
    best_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model_checkpoint.keras',
        monitor='val_rmsle',
        save_best_only=True,
        mode='min'
    )

    # Search with optimized callbacks
    tuner.search(
        X_train, y_train,  # Changed from y_train_log to y_train
        epochs=100,
        batch_size=128,  # Increased from 64
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_rmsle',
                patience=5,  # Reduced patience to prevent overfitting
                restore_best_weights=True,
                mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_rmsle',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                mode='min'
            ),
            best_model_checkpoint,
            csv_logger
        ],
        verbose=1
    )

    # Get and save best model
    best_hps = tuner.get_best_hyperparameters(1)[0]
    logger.info(f"Best hyperparameters: {best_hps.values}")

    # Remove the redundant code for training the best model again
    # Since we are saving the best model during tuning, this is no longer necessary
    # ...commented out code...

    # Load and evaluate the best model
    best_model = tf.keras.models.load_model('best_model_checkpoint.keras', custom_objects={'rmsle': rmsle})
    val_pred_log = best_model.predict(X_val)
    val_pred = np.expm1(val_pred_log.flatten())
    final_rmsle = np.sqrt(np.mean(np.square(np.log1p(val_pred) - np.log1p(y_val))))
    logger.info(f"Final validation RMSLE: {final_rmsle:.4f}")

except Exception as e:
    logger.error(f"Error during model training: {str(e)}")
    raise
