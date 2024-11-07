import logging
import pandas as pd
from src.ingest_data import DataLoader
from src.data_preprocessing import DataPreprocessor, BasicPreprocessingStrategy
from src.data_sampling import DataSampler
from src.vectorization import TfidfVectorization 
from scipy.sparse import csr_matrix

def get_data_for_test(table_name: str="customer_reviews", db_uri: str = "postgresql://postgres:3333@localhost:5432/NLP") -> str:
    """
    Loads, preprocesses, samples, and vectorizes data for testing and returns it as a JSON string.
    
    Parameters:
        table_name (str): Name of the PostgreSQL table to load data from.
        db_uri (str): Database URI for connecting to PostgreSQL. Default is set for local setup.
    
    Returns:
        str: JSON string of the vectorized test data in "split" orientation.
    """

    try:
        logging.info("Loading data from PostgreSQL table.")
        data_loader = DataLoader(db_uri)
        data_loader.load_data(table_name)
        df = data_loader.get_data()

        if df.empty:
            logging.warning("Loaded data is empty. Check database table content.")

        logging.info(f"Data loaded successfully with {len(df)} records.")

        logging.info("Starting data preprocessing.")
        preprocessor = DataPreprocessor(strategy=BasicPreprocessingStrategy())
        df = preprocessor.preprocess(df)
        
        if df.empty:
            logging.warning("Preprocessed data is empty.")

        logging.info(f"Data preprocessing complete. Number of records after preprocessing: {len(df)}.")

        logging.info("Sampling data for testing.")
        sampler = DataSampler(df)
        df = sampler.sample_data(df)

        if "label" in df.columns:
            df.drop(columns=["label"], inplace=True)

        logging.info(f"Final data shape for inference: {df.shape}")

        logging.info("Starting vectorization of the review text.")
        vectorizer = TfidfVectorization()  
        vectorized_data = vectorizer.fit_transform(df["review_text"]) 

        logging.info("Vectorization complete. Shape of vectorized data: {}".format(vectorized_data.shape))
        
        # Convert vectorized data to DataFrame or keep it as CSR matrix for model input
        vectorized_df = pd.DataFrame(vectorized_data.toarray())

        # Convert to JSON for return
        result = vectorized_df.sample(100).to_json(orient="split")
        return result
    
    except Exception as e:
        logging.error(f"Error in get_data_for_test: {e}")
        raise e
