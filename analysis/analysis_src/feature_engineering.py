import nltk
import numpy as np
import pandas as pd


# Download the necessary NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

class FeatureEngineering:
    
    def __init__(self, data):
        self.data = data
        
    def feature_extraction(self):
        data_copy = self.data.copy()
        
        # Length of review text
        data_copy['length_of_review'] = data_copy['review_text'].apply(len)
        
        # Total number of words in review
        data_copy['total_number_of_words'] = data_copy['review_text'].apply(lambda text: len(nltk.word_tokenize(text)))
        
        # Number of unique words in review
        data_copy['number_of_unique_words'] = data_copy['review_text'].apply(lambda text: len(set(nltk.word_tokenize(text))))
        
        # Ratio of unique words to total words
        data_copy['unique_to_total_word_ratio'] = data_copy['number_of_unique_words'] / data_copy['total_number_of_words']
        
        return data_copy
