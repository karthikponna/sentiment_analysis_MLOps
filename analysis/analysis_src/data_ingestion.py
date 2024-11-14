import logging
import pandas as pd
from sqlalchemy import create_engine, exc

class DataLoader:
    """A class to load customer reviews data from a PostgreSQL database.
    
    Attributes:
        db_uri:str
            The database URI to connect to the PostgreSQL database.
            
    Methods:
        load_data(table_name):
            Loads the data from the specified table and returns a DataFrame.
    """
    def __init__(self, db_uri:str):
        """
        Initializes the DataIngestor with the database URI.
        
        Parameters:
            db_uri: str
                The URI to connect to the PostgreSQL database.
        """
        self.db_uri=db_uri
        self.engine=create_engine(self.db_uri)
        self.data=None

    def load_data(self,table_name:str)-> pd.DataFrame:
        """
        Loads the data from the specified table in the PostgreSQL database.
        
        Args:
            table_name: Name of the table to read from.
        
        Returns:
            pd.DataFrame: A DataFrame containing the customer reviews.
        """
        query = "SELECT * FROM "+table_name
        try:
            self.data = pd.read_sql(query, self.engine)
            logging.info(f"Successfully loaded data from the table {table_name}.")
            return self.data
        except exc.SQLAlchemyError as e:
            raise e
        
    def get_data(self) -> pd.DataFrame:
        """
        Returns the data that was loaded into the class instance.

        Returns:
            pd.DataFrame: Data from the table.
        """
        if self.data is not None:
            return self.data
        else:
            raise ValueError("No data loaded yet. Please run load_data() first.")