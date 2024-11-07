import logging
import pandas as pd
from src.ingest_data import DataLoader

from zenml import step

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=False)
def ingest_data(
    table_name:str,
) -> pd.DataFrame:
    """
    Ingests data from the specified PostgreSQL table.

    Parameters:
        table_name : str
            The name of the table containing the customer reviews.
        for_predict : bool
            A flag indicating if the data is being prepared for prediction.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded customer reviews.
    """

    logging.info("Started data ingestion process.")
    
    try:

        data_loader = DataLoader("postgresql://postgres:3333@localhost:5432/NLP")
        data_loader.load_data(table_name)
        df = data_loader.get_data()

        if df.empty:
            logging.warning("No data was loaded. Check the table name or the database content.")
        else:
            logging.info(f"Data ingestion completed. Number of records loaded: {len(df)}.")
    
        return df
    except Exception as e:
        logging.error(f"Error while reading data from {table_name}: {e}")
        raise e 

