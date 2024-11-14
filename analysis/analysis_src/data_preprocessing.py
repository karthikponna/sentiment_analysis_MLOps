import logging
import pandas as pd
import numpy as np
import nltk
import re
import string
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')


# Abstract Base Class for Preprocessing Strategy
# -----------------------------------------------
# This class defines a common interface for different preprocessing strategies.
# Subclasses must implement the preprocess method.
class PreprocessingStrategy(ABC):
    @abstractmethod
    def data_preprocessing(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to preprocess the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame to be processed.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        pass

# Concrete Strategy for Basic Preprocessing
# ---------------------------------------------
# This strategy implements basic preprocessing and text_preprocessing steps for customer reviews data.
class BasicPreprocessingStrategy(PreprocessingStrategy):
    def __init__(self):
        """
        Initializes the BasicPreprocessingStrategy with stop words and lemmatizer.

        Attributes:
            my_stopword (set): A set of English stop words.
            my_lemmatizer (WordNetLemmatizer): An instance of the WordNetLemmatizer for lemmatization.
        """
        self.my_stopword = set(stopwords.words('english'))
        self.my_lemmatizer = WordNetLemmatizer()

    def data_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Started basic preprocessing of the data.")

        # Check if 'score' is in columns
        if 'score' not in df.columns:
            logging.error("The 'score' column is missing in the DataFrame.")
            return df

        df = df[['review_text', 'score']]  # Only keep review_text and score columns

        # Drop NA values and reset index
        df = df.dropna().reset_index(drop=True)

        # Changing 'score' column to int type
        df['score'] = df['score'].astype(int)

        # Remove neutral reviews (Score 3)
        df = df[df['score'] != 3]

        # Label the reviews (Positive - 1, Negative - 0)
        df['label'] = np.where(df['score'] >= 4, 1, 0)

        # Dropping 'score' column
        df = df.drop(columns=['score'])

        # Apply text preprocessing on review_text column
        df["review_text"] = df["review_text"].apply(self.text_preprocessing)

        logging.info(f"Basic preprocessing completed. Number of records after preprocessing: {len(df)}.")

        return df 
    
    def text_preprocessing(self, text):
        """
        Apply a series of text preprocessing steps to the given text.

        Parameters:
            text (str): The input text to be preprocessed.

        Returns:
            str: The processed text.
        """
        text = self.lower_text(text)
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.replace_special_character_to_string_equivalent(text)
        text = self.expand_contractions(text)
        text = self.remove_non_alpha(text)
        text = self.remove_extra_spaces(text)
        text = self.remove_stopwords(text)
        text = self.text_lemmatization(text)
        return text
    
    def lower_text(self, text):
        """
        Converts the input text to lowercase.

        Parameters:
            text (str): The input text to convert.

        Returns:
            str: The lowercase version of the input text.
        """
        text = text.lower()
        return text
    
    def remove_html_tags(self, text):
        """
        Removes HTML tags from the input text.

        Parameters:
            text (str): The input text potentially containing HTML tags.

        Returns:
            str: The text without HTML tags.
        """
        text = BeautifulSoup(text, "html.parser").get_text()
        return text
    
    def remove_urls(self, text):
        """
        Removes URLs from the input text.

        Parameters:
            text (str): The input text potentially containing URLs.

        Returns:
            str: The text without URLs.
        """
        text = re.sub(r"http\S+","", text)
        return text
    
    def replace_special_character_to_string_equivalent(self,text):
        """
        Replaces special characters in the input text with their string equivalents.

        Parameters:
            text (str): The input text containing special characters.

        Returns:
            str: The text with special characters replaced by their equivalents.
        """
        replacements = {
            '%':"percent", 
            '$':"dollar",
            '₹':"rupee",
            '€':"euro",
            '@':"at",
        }
        for char,word in replacements.items():
            text = text.replace(char,word)
        return text
    
    def expand_contractions(self, text):
        """
        Expands contractions in the input text to their full form.

        Parameters:
            text (str): The input text containing contractions.

        Returns:
            str: The text with contractions expanded.
        """
        contractions = {
            "won't":"will not",
            "can't":"cannot",
            "n't":"not",
            "'re":"are",
            "'s":"is",
            "'d":"would",
            "'ll":"will",
            "'t":"not",
            "'ve":"have",
            "'m":"am",
        }
        for contraction, expand in contractions.items():
            text = re.sub(contraction, expand, text)
        return text
    
    def remove_non_alpha(self,text):
        """
        Removes non-alphabetical characters from the input text.

        Parameters:
            text (str): The input text to process.

        Returns:
            str: The text with non-alphabetical characters removed.
        """
        words = nltk.word_tokenize(text)
        words = [re.sub('[^A-Za-z]','',word) for word in words]
        return ' '.join(words)
    
    def remove_extra_spaces(self,text):
        """
        Removes extra spaces from the input text.

        Parameters:
            text (str): The input text to process.

        Returns:
            str: The text with extra spaces removed.
        """
        text = re.sub(r'\s+',' ', text).strip()
        return text 
    
    def remove_stopwords(self,text):
        """
        Removes stopwords from the input text.

        Parameters:
            text (str): The input text from which to remove stopwords.

        Returns:
            str: The text with stopwords removed.
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.my_stopword]
        return ' '.join(filtered_words)
    
    def text_lemmatization(self, text):
        """
        Lemmatizes the words in the input text.

        Parameters:
            text (str): The input text to lemmatize.

        Returns:
            str: The lemmatized text.
        """
        words = nltk.word_tokenize(text)
        lemmatized_words = [self.my_lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    

    
# Context Class for Data Preprocessing
# --------------------------------------
# This class uses a PreprocessingStrategy to preprocess the data.
class DataPreprocessor:
    def __init__(self, strategy:PreprocessingStrategy):
        """
        Initializes the DataPreprocessor with the DataFrame and a strategy.

        Parameters:
            df (pd.DataFrame): The DataFrame containing customer reviews data.
            strategy (PreprocessingStrategy): The strategy for preprocessing.
        """
        self._strategy=strategy

    def set_strategy(self, strategy:PreprocessingStrategy):
        """
        Sets a new strategy for the DataPreprocessor.

        Parameters:
            strategy (PreprocessingStrategy): The new strategy to be used for preprocessing.
        """
        logging.info("Switching preprocessing strategy")
        self._strategy=strategy

    def preprocess(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Executes the preprocessing using the current strategy.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        logging.info("Preprocessing data using the selected strategy.")
        return self._strategy.data_preprocessing(df)
    

# Example usage
if __name__ == "__main__":
    
    # # Example DataFrame (replace with actual data loading)
    # df = pd.DataFrame({
    #     'review_text': ['Good product! Highly recommend.', 'Just okay.', 'Worst product ever!'],
    #     'score': [5, 4, 1],
    #     'label': [1, 1, 0]
    # })

    # # Initialize data preprocessor with a specific strategy
    # strategy = BasicPreprocessingStrategy()
    # preprocessor = DataPreprocessor(strategy)
    # preprocessed_data = preprocessor.preprocess(df)
    # print(preprocessed_data)

    pass
