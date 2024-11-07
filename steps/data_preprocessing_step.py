import logging
from typing import Tuple
import pandas as pd
from src.data_preprocessing import DataPreprocessor, BasicPreprocessingStrategy

from zenml import step

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step
def data_preprocessing_step(df:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data using DataPreprocessor and a chosen strategy.

    Parameters:
        df : pd.DataFrame
            The DataFrame containing reviews and their scores.

    Returns:
        pd.DataFrame: A preprocessed DataFrame containing the reviews and labels.
    """
    logging.info("Start data preprocessing step.")
    preprocessor = DataPreprocessor(strategy=BasicPreprocessingStrategy())

    processed_df = preprocessor.preprocess(df)

    if processed_df.empty:
        logging.warning("The preprocessing resulted in an empty DataFrame. Check the data for issues.")
    else:
        logging.info(f"Preprocessing completed. Number of records after preprocessing: {len(processed_df)}")

    return processed_df



 