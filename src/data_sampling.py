import pandas as pd
import logging

class DataSampler:
    """A class to sample customer reviews data for model training.
    
    Attributes:
        df: pd.DataFrame
            The DataFrame containing labeled customer reviews data.
    
    Methods:
        sample_data() -> pd.DataFrame:
            Samples 100,000 reviews (50,000 positive and 50,000 negative).
    """
    def __init__(self, df:pd.DataFrame):
        """
        Initializes the DataSampler with the DataFrame.
        
        Parameters:
            df: pd.DataFrame
                The DataFrame containing labeled customer reviews data.
        """
        self.df = df

    def sample_data(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Samples 100,000 reviews by shuffling the DataFrame 
        and selecting 50,000 positive and 50,000 negative reviews.
        
        Returns:
            pd.DataFrame: A DataFrame containing the sampled reviews.
        """
        logging.info("Started sampling data for model training.")

        #Shuffle the data
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Select 50,000 negative reviews (label 0)
        negative_reviews = df_shuffled[df_shuffled["label"]==0][:3000]

        #select 50,000 positive reviews (label 1)
        positive_reviews = df_shuffled[df_shuffled["label"]==1][:3000]

        # Combine the selected reviews
        sampled_data = pd.concat([negative_reviews, positive_reviews], ignore_index=True)

        logging.info(f"Sampling completed. Number of records after sampling: {len(sampled_data)}.")

        return sampled_data