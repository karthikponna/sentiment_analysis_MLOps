
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