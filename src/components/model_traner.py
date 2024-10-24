import os
import sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import *

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore")



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):

        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {    
                'Logistic Regression' : LogisticRegression(),
                'Decision Tree Regressor' : DecisionTreeClassifier(),
                "Naib Bias" : GaussianNB(),
                'SVC' : SVC(),
                'KNN': KNeighborsClassifier(),
                'Random Forest Classifier' : RandomForestClassifier(),
                'Adaboosting' : AdaBoostClassifier(),
                'GradientBoosting' : GradientBoostingClassifier()
            }

            logging.info("evaluating model...")
            model_report = {}
            for i in range(len(models)):
                print(i)
                model = list(models.values())[i]
                # print(model)
                # Train model
                
                model.fit(X_train,y_train)
                print(model)
                # Predict Testing data
                y_test_pred =model.predict(X_test)
                print(y_test_pred[1])
                # Get R2 scores for train and test data
                #train_model_score = r2_score(ytrain,y_train_pred)
                test_model_score = accuracy_score(y_test,y_test_pred)

                model_report[list(models.keys())[i]] =  test_model_score
                logging.info("model evaluated..")

            # model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , accuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )


        except Exception as e:
            CustomException(e,sys)