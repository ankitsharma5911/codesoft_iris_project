import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion

obj = DataIngestion()
train_data_path,test_data_path = obj.initialize_data_ingestion()

