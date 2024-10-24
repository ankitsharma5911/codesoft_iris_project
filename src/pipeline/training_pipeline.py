import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_traner import ModelTrainer

obj = DataIngestion()
train_data_path,test_data_path = obj.initialize_data_ingestion()

print(train_data_path,test_data_path)

transformation = DataTransformation()
train_arr,test_arr = transformation.initiate_data_transformation(train_data_path,test_data_path)

trainer = ModelTrainer()
trainer.initate_model_training(train_arr,test_arr)
