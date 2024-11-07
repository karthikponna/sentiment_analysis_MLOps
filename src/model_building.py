import logging
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
import joblib

# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train:pd.DataFrame, y_train:pd.Series, fine_tuning:bool = False) -> Any:
        """
        Abstract method to build and train a model.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.
            fine_tuning (bool): Flag to indicate if fine-tuning should be performed.

        Returns:
            Any: A trained scikit-learn model instance.
        """
        pass

# Concrete Strategy for Logistic Regression
class LogisticRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool = False) -> Any:
        """
        Trains a Logistic Regression model on the provided training data.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.
            fine_tuning (bool): Not applicable for Logistic Regression, defaults to False.

        Returns:
            LogisticRegression: A trained Logistic Regression model.
        """
        logging.info("Training the Logistic Regression Model.")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train,y_train)
        joblib.dump(model, "/home/karthikponna/karthik/sentiment_analysis_mlops_project_1/sentiment_analysis_MLOps/saved_models/logistic.pkl")
        logging.info("Logistic Regression training completed.")
        return model
    
# Concrete Strategy for XGBoost
class XGBoostStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool = False) -> Any:
        """
        Trains an XGBoost model on the provided training data, optionally with fine-tuning.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.
            fine_tuning (bool): Flag to indicate if fine-tuning should be performed.

        Returns:
            XGBClassifier: A trained XGBoost model (either fine-tuned or default).
        """
        if fine_tuning:
            logging.info("Started fine-tuning the XGBoost model.")
            params = {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [2, 4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.5, 0.7, 0.8, 1.0],
                "colsample_bytree": [0.5, 0.7, 1.0],
            }
            
            xgb_model = XGBClassifier()
            clf = RandomizedSearchCV(xgb_model, params, cv=5, n_jobs=-1)
            clf.fit(X_train, y_train)
            joblib.dump(clf, "/home/karthikponna/karthik/sentiment_analysis_mlops_project_1/sentiment_analysis_MLOps/saved_models/xgb_fine_tuned.pkl")
            logging.info("Finished Hyperparameter search for XGBoost.")
            return clf
        
        else:
            logging.info("Started training the XGBoost model.")
            model = XGBClassifier(
                learning_rate=0.3,
                max_depth=8,
                min_child_weight=1,
                n_estimators=50,
                random_state=0,
            )
            
            model.fit(X_train, y_train)
            joblib.dump(model, "/home/karthikponna/karthik/sentiment_analysis_mlops_project_1/sentiment_analysis_MLOps/saved_models/xgb.pkl")
            logging.info("Completed training the XGBoost model.")
            return model
        
# Concrete Strategy for SVM
class SVCStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool = False) -> Any:
        """
        Trains a Support Vector Classifier model on the provided training data, optionally with fine-tuning.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.
            fine_tuning (bool): Flag to indicate if fine-tuning should be performed.

        Returns:
            SVC: A trained SVC model (either fine-tuned or default).
        """
        if fine_tuning:
            logging.info("Started fine-tuning the SVM model.")
            params = {
                "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            }
            svm = SVC()
            clf = RandomizedSearchCV(svm, params, cv=5, n_jobs=-1)
            clf.fit(X_train, y_train)
            joblib.dump(clf, "/home/karthikponna/karthik/sentiment_analysis_mlops_project_1/sentiment_analysis_MLOps/saved_models/svm_fine_tuned.pkl")
            logging.info("Finished Hyperparameter search for SVM.")
            return clf
        
        else:
            logging.info("Started training the SVM model.")
            model = SVC(C=1.0, gamma='scale')
            model.fit(X_train, y_train)
            joblib.dump(model, "/home/karthikponna/karthik/sentiment_analysis_mlops_project_1/sentiment_analysis_MLOps/saved_models/svm.pkl")
            logging.info("Completed training the SVM model.")
            return model
            
# Concrete Strategy for Naive Bayes
class NaiveBayesStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool = False) -> Any:
        """
        Trains a Naive Bayes model on the provided training data.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.
            fine_tuning (bool): Not applicable for Naive Bayes, defaults to False.

        Returns:
            MultinomialNB: A trained Naive Bayes model.
        """
        logging.info("Training the Naive Bayes model.")
        model = MultinomialNB()
        model.fit(X_train, y_train)
        joblib.dump(model, "/home/karthikponna/karthik/sentiment_analysis_mlops_project_1/sentiment_analysis_MLOps/saved_models/NBayes.pkl")
        logging.info("Completed training the Naive Bayes model.")
        return model
    
# Concrete Strategy for Random Forest
class RandomForestStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool = False) -> Any:
        """
        Trains a Random Forest model on the provided training data.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.
            fine_tuning (bool): Not applicable for Random Forest, defaults to False.

        Returns:
            RandomForestClassifier: A trained Random Forest model.
        """
        logging.info("Training the Random Forest model.")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, "/home/karthikponna/karthik/sentiment_analysis_mlops_project_1/sentiment_analysis_MLOps/saved_models/rf.pkl")
        logging.info("Completed training the Random Forest model.")
        return model
    
# Context Class for Model Building Strategy
class ModelBuilder:
    def __init__(self, strategy:ModelBuildingStrategy):
        """
        Initializes the ModelBuildingStrategy with the X_train, y_train, fine_tuning and a strategy.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.
            fine_tuning (bool): Flag to indicate if fine-tuning should be performed.
        """
        self._strategy = strategy

    def set_strategy(self, strategy:ModelBuildingStrategy):
        """
        Set the model building strategy.

        Parameters:
            strategy (ModelBuildingStrategy): The strategy to set.
        """
        self._strategy = strategy

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool = False) -> Any:
        """
        Train the model using the set strategy.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.
            fine_tuning (bool): Flag to indicate if fine-tuning should be performed.

        Returns:
            Any: A trained model instance from the chosen strategy.
        """
        return self._strategy.build_and_train_model(X_train, y_train, fine_tuning)


# Example usage
if __name__ == "__main__":
    # import numpy as np
    # from sklearn.model_selection import train_test_split
    # from sklearn.datasets import make_classification

    # # Configure logging
    # logging.basicConfig(level=logging.INFO)

    # # Generate synthetic data for demonstration purposes
    # X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Initialize the ModelBuilder with different strategies
    # # Logistic Regression
    # logging.info("Training Logistic Regression Model")
    # logistic_strategy = LogisticRegressionStrategy()
    # logistic_builder = ModelBuilder(logistic_strategy)
    # logistic_model = logistic_builder.train(X_train, y_train)
    # logging.info("Logistic Regression Model Trained and Saved.")

    # # XGBoost with fine-tuning
    # logging.info("Training XGBoost Model with Fine-Tuning")
    # xgb_strategy = XGBoostStrategy()
    # xgb_builder = ModelBuilder(xgb_strategy)
    # xgb_model = xgb_builder.train(X_train, y_train, fine_tuning=True)
    # logging.info("XGBoost Model (Fine-Tuned) Trained and Saved.")

    # # SVM with fine-tuning
    # logging.info("Training SVM Model with Fine-Tuning")
    # svm_strategy = SVCStrategy()
    # svm_builder = ModelBuilder(svm_strategy)
    # svm_model = svm_builder.train(X_train, y_train, fine_tuning=True)
    # logging.info("SVM Model (Fine-Tuned) Trained and Saved.")

    # # Naive Bayes
    # logging.info("Training Naive Bayes Model")
    # nb_strategy = NaiveBayesStrategy()
    # nb_builder = ModelBuilder(nb_strategy)
    # nb_model = nb_builder.train(X_train, y_train)
    # logging.info("Naive Bayes Model Trained and Saved.")

    # # Random Forest
    # logging.info("Training Random Forest Model")
    # rf_strategy = RandomForestStrategy()
    # rf_builder = ModelBuilder(rf_strategy)
    # rf_model = rf_builder.train(X_train, y_train)
    # logging.info("Random Forest Model Trained and Saved.")

    # # Example of using models on test data (Optional)
    # test_data_sample = X_test[:5]  # Take first 5 examples for testing
    # logging.info("Logistic Regression Prediction: %s", logistic_model.predict(test_data_sample))
    # logging.info("XGBoost Prediction: %s", xgb_model.predict(test_data_sample))
    # logging.info("SVM Prediction: %s", svm_model.predict(test_data_sample))
    # logging.info("Naive Bayes Prediction: %s", nb_model.predict(test_data_sample))
    # logging.info("Random Forest Prediction: %s", rf_model.predict(test_data_sample))
    
    pass