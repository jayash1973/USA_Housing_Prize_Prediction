import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.dta_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''    
        try:
            numerical_columns = ['date','price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated']
            categorical_columns = [
                'street',
                'city',
                'statezip',
                'country'
            ]
            
            num_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler())
                ]
            )
            
            logging.info(f"numerical columns: {numerical_columns}")
            
            logging.info(f'categorical columns: {categorical_columns}')
            
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info('read train and test data completed')
            
            logging.info('obtaining preprocessing object')
            
            preprocessinf_obj=self.get_data_transformer_object()
            
            
        except:
            pass