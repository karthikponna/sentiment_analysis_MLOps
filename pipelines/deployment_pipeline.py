import os
import json
import numpy as np
import pandas as pd 
from .utils import get_data_for_test
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
  
#Inference Pipeline
class MLFlowDeploymentLoaderStepParameters(BaseParameters):
  pipeline_name:str
  step_name:str
  running:bool=True

@step(enable_cache=False)
def dynamic_importer()->str:
  data=get_data_for_test()
  return data  

@step(enable_cache=False)
def prediction_service_loader(
  pipeline_name: str,
  pipeline_step_name: str,
  running: bool=True,
  model_name: str="model",
) -> MLFlowDeploymentService:
  """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
  
  # get the MLflow model deployer stack component
  mlflow_model_deployer_component=MLFlowModelDeployer.get_active_model_deployer()

  # fetch existing services with same pipeline name, step name and model name
  existing_services=mlflow_model_deployer_component.find_model_server(
  pipeline_name=pipeline_name,
  pipeline_step_name=pipeline_step_name,
  model_name=model_name,
  running=running,
)
  if not existing_services:
      raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."      
          )
  
  print("Existing Services:", existing_services)
  print(type(existing_services))
  return existing_services[0]

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
  """Run an inference request against a prediction service"""

  service.start(timeout=10)  # should be a NOP if already started
  data = json.loads(data)
  data.pop("columns")
  data.pop("index")
  
  
  try:
    # Ensure the data is correctly structured with 4995 features per sample
    input_data = np.array(data["data"])  # Shape should be (100, 4995)
    print(f"Input data to model: {input_data.shape}")
        
    # Send the data to the prediction service
    prediction = service.predict(input_data)
    return prediction
  
  except Exception as e:
    print(f"Prediction error: {str(e)}")

@pipeline(enable_cache=False, settings={"docker":docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name:str):

  # Load batch data for inference
  data=dynamic_importer()
  
  # Load the deployed model service
  service=prediction_service_loader(
    pipeline_name=pipeline_name,
    pipeline_step_name=pipeline_step_name,
    running=False,
  )

  # Run predictions on the batch data
  prediction=predictor(service=service,data=data)
  return prediction
