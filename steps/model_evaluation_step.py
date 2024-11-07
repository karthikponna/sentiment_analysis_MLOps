import mlflow
import logging
from typing import Dict, Any, List
import pandas as pd
from sklearn.base import ClassifierMixin
from scipy.sparse import csr_matrix

from src.model_evaluation import ModelEvaluator, ClassificationModelEvaluationStrategy

from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker

logger = get_logger(__name__)

experiment_tracker=Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )

@step(experiment_tracker=experiment_tracker.name)
def model_evaluation_step(
    model:ClassifierMixin, X_test:csr_matrix, y_test:pd.Series
) -> Dict[str, float]:
    """
    Step to evaluate a classification model.

    Parameters:
        model (ClassifierMixin): The trained classification model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    try:
        
        evaluator = ModelEvaluator(ClassificationModelEvaluationStrategy())
        metrics = evaluator.evaluate(model, X_test, y_test)

        #log metrics to mlflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            
        return metrics
        
    except Exception as e:
        logging.error(f"Error in evaluating the model: {e}")
        raise e


# Example usage
if __name__ == "__main__":
    
    pass