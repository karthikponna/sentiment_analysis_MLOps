import logging
import pandas as pd
from src.data_sampling import DataSampler

from zenml import step

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step
def data_sampling_step(df:pd.DataFrame) -> pd.DataFrame:
    """
    Samples the customer reviews data to create a balanced dataset for model training.

    Parameters:
        df : pd.DataFrame
            The DataFrame containing labeled customer reviews data.

    Returns:
        pd.DataFrame: A balanced DataFrame with 100,000 sampled reviews 
                      (50,000 positive and 50,000 negative).
    """
    logging.info("Started data sampling step.")

    sampler = DataSampler(df)

    sampled_df = sampler.sample_data(df)

    if sampled_df.empty:
        logging.warning("The sampling step resulted in an empty DataFrame. Check the data for issues.")
    else:
        logging.info(f"Sampling step completed. Number of records after sampling: {len(sampled_df)}.")

    return sampled_df
