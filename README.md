zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set




chatgpt (for model selection)

so i said earlier i'm building a project named sentiment analysis where i'm using logistic_regression, xgboost, svc, multinomialNB, random_forest. so these are the respective metrics 


Key Metrics Breakdown

Accuracy (0.8875): Indicates that your model correctly predicted the sentiment in about 88.75% of cases. This is solid for a classification task, especially if accuracy is relatively close to other key metrics.

Recall (0.8838): Shows that the model captured approximately 88.4% of all true positive cases. A high recall suggests that the model is doing well at identifying positive cases and minimizing false negatives, which is useful in sentiment analysis where you want to ensure you capture most of the positive sentiments.

Precision (0.8883): Indicates that 88.83% of positive predictions are correct. High precision, paired with high recall, shows that the model is good at distinguishing between positive and negative cases, with relatively low false positives.

F1 Score (0.8860): This metric balances precision and recall, and here, it’s very close to both values, which implies a good balance between capturing all positives and not over-predicting them.

ROC AUC (0.8875): Measures the model’s ability to distinguish between classes (positive and negative sentiment). This score is strong, suggesting that the model performs well in ranking positive cases higher than negative ones.

Training Metrics (Accuracy, Recall, Precision, F1): Training metrics are all around 0.99, which is higher than your test metrics. This could indicate slight overfitting, as the model is performing nearly perfectly on the training data but with a slight drop on unseen data. However, the drop isn’t severe, so it’s generally acceptable.




solve this error

Training the Random Forest model.
`2024/11/02 11:06:10 WARNING mlflow.sklearn: Failed to log training dataset information to MLflow Tracking. Reason: 'Series' object has no attribute 'toarray'`

<!-- doubt in -->
1. deployment_pipeline.py -> deploy_decision=True 
2. @click.option(
    "--min-accuracy",
    default=0.0,
    help="Minimum accuracy required to deploy the model",
)               ####try changing the default to 0.5 and run



<!-- deployment_pipeline.py file -->

import os
import json
import numpy as np
import pandas as pd 
# from .utils import get_data_for_test
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import(MLFlowModelDeployer,)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from pipelines.training_pipeline import ml_pipeline


docker_settings=DockerSettings(required_integrations=[MLFLOW])


#Continuous Deployment Pipeline
class DeploymentTriggerConfig(BaseParameters):
  """Class for configuring deployment trigger"""
  min_accuracy: float=0.5


@step
def deployment_trigger(
  accuracy:float,
  config: DeploymentTriggerConfig,
) -> bool:
  return accuracy>config.min_accuracy


@pipeline(enable_cache=False, settings={"docker":docker_settings})
def continuous_deployment_pipeline(
  min_accuracy:float=0.5,
  workers: int=1,
  timeout: int=DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
  """Run a training job and deploy an MLflow model deployment."""

  # Run the training pipeline
  trained_model = ml_pipeline()

  # (Re)deploy the trained model
  mlflow_model_deployer_step(
      model=trained_model,
      deploy_decision=True,
      workers=workers,
      timeout=timeout,
    )


<!-- run_deployment.py file -->
import click
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

@click.command()
def main():
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    # Run the pipeline
    run = ml_pipeline()

    # You can uncomment and customize the following lines if you want to retrieve and inspect the trained model:
    # trained_model = run["model_building_step"]  # Replace with actual step name if different
    # print(f"Trained Model Type: {type(trained_model)}")

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )

if __name__=="__main__":
    main()

<!-- 
predictor step from deployment_pipeline.py file -->


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
  """Run an inference request against a prediction service"""

  service.start(timeout=10)  # should be a NOP if already started
  data = json.loads(data)
  
  # Since data is vectorized (not in raw text form), we don't need to extract columns
  try:
    # Directly use the data for prediction (this is already vectorized)
    data = np.array(data["data"])
    
    prediction = service.predict(data)
    return prediction
  
  except Exception as e:
    print(f"Prediction error: {str(e)}")




@@@@ i changed max_features and dtype in tfidfvectorization in vectorization.py 


##### model_building_step.py file #####


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



## training_pipeline.py file

from steps.data_ingestion_step import ingest_data
from steps.data_preprocessing_step import data_preprocessing_step
from steps.data_sampling_step import data_sampling_step
from steps.data_splitter_step import data_splitter_step
from steps.vectorization_step import vectorization_step
from steps.model_building_step import model_building_step
from steps.model_evaluation_step import model_evaluation_step
from zenml import Model, pipeline

@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="customer_reviews_predictor"
    ),
)
def ml_pipeline():

    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = ingest_data("customer_reviews")

    df_preprocessed = data_preprocessing_step(raw_data)
    
    df_sampled = data_sampling_step(df_preprocessed)
    
    X_train, X_test, y_train, y_test = data_splitter_step(df_sampled, target_column="label")

    tf_X_train, tf_X_test = vectorization_step(X_train, X_test)

    model = model_building_step(X_train=tf_X_train, y_train=y_train, method="svc", fine_tuning=False)

    evaluator = model_evaluation_step(model=model, X_test=tf_X_test, y_test=y_test)

    return evaluator

if __name__=="__main__":
    run = ml_pipeline()