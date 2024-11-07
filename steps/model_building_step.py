
import mlflow
import logging
import pandas as pd
import numpy as np
from typing import List, Annotated
from scipy.sparse import csr_matrix
from sklearn.base import ClassifierMixin
from src.model_building import (
    LogisticRegressionStrategy,
    XGBoostStrategy,
    SVCStrategy,
    NaiveBayesStrategy,
    RandomForestStrategy,
    ModelBuilder
)
from materializers.custom_materializer import SVCMaterializer, Int64Materializer

from zenml import ArtifactConfig, step
from zenml.client import Client

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

if experiment_tracker is None:
    raise ValueError("Experiment tracker is not initialized. Please ensure ZenML is set up correctly.")

from zenml import Model

model = Model(
    name="customer_reviews_predictor",
    version=None,
    license="Apache 2.0",
    description="Reviews predictor model for customer reviews",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train:csr_matrix, y_train:pd.Series, method:str, fine_tuning:bool = False
) -> Annotated[ClassifierMixin, ArtifactConfig(name="trained_model", is_model_artifact=True, materializer=SVCMaterializer)]:
    """
    Model building step using ZenML with multiple model options and MLflow tracking.

    Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.
        method (str): Model selection method, e.g., 'logistic_regression', 'xgboost', 'svm', 'naive_bayes', 'random_forest'.
        fine_tuning (bool): Flag to indicate if fine-tuning should be performed, only applicable to certain models.

    Returns:
        Trained model instance.
    """
    logging.info(f"Building model using method: {method}")
    
    # Choose the appropriate strategy based on the method
    if method == "logistic_regression":
        strategy = LogisticRegressionStrategy()
        logging.info("Selected Logistic Regression Strategy.")
    
    elif method == "xgboost":
        strategy = XGBoostStrategy()
        logging.info("Selected XGBoost Strategy.")

    elif method == "svc":
        strategy = SVCStrategy()
        logging.info("Selected SVM Strategy.")

    elif method == "naive_bayes":
        strategy = NaiveBayesStrategy()
        logging.info("Selected Naive Bayes Strategy.")

    elif method == "random_forest":
        strategy = RandomForestStrategy()
        logging.info("Selected Random Forest Strategy.")

    else:
        raise ValueError(f"Unknown method '{method}' selected for model training.")
    
    # Initialize ModelBuilder with the selected strategy
    model_builder = ModelBuilder(strategy)

    # Start an MLflow run to log the training process
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Enable autologging to automatically log model parameters, metrics, and artifacts
        mlflow.sklearn.autolog()

        # Log training data (optional but useful for tracking)
        mlflow.log_param("X_train_shape", X_train.shape)
        mlflow.log_param("y_train_shape", np.array(y_train).shape)

        # Train the model with or without fine-tuning
        logging.info("Started model training.")
        trained_model = model_builder.train(X_train, np.array(y_train), fine_tuning=fine_tuning)
        logging.info("Model training completed.")

    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise e
    
    finally:
        #End the mlflow run
        mlflow.end_run()

    return trained_model
