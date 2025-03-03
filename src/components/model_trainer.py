import os
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor 
from dataclasses import dataclass
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb 
from xgboost import XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object
# from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression(solver='liblinear'),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Naive Bayes": GaussianNB(),
                "Support Vector Machine": SVC(),
                "GBM": GradientBoostingClassifier(),
                "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss'),
                "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.5:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            
            predicted = best_model.predict(X_test)
            # accuracy_score = best_model.score(y_test, predicted)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy





            
        except Exception as e:
            raise CustomException(e,sys)
        