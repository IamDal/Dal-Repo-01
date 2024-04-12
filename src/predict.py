import os
import pandas as pd
import numpy as np

import config
from train import TrainModel
from data import Preprocess
import matplotlib.pyplot as plt 
import seaborn as sns 
import pickle

import sys
sys.path.append('/opt/conda/lib/python3.12/site-packages')

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
# space to delete
# Import model selection
from sklearn.model_selection import train_test_split

# Import accuracy metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Import pipeline and preprocessing imputers and encoders
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class Customer:
    def __init__(self, CustomerData):
        self.CustomerId = CustomerData[0]
        self.Surname = CustomerData[1]
        self.Creditscore = CustomerData[2]
        self.Geography = CustomerData[3]
        self.Gender = CustomerData[4]
        self.Age = CustomerData[5]
        self.Tenure = CustomerData[6]
        self.Balance = CustomerData[7]
        self.NumOfProducts = CustomerData[8]
        self.HasCrCard = CustomerData[9]
        self.IsActiveMember = CustomerData[10]
        self.EstimatedSalary = CustomerData[11]

    def CustomerDataFrame(self):
        data = {
                'RowNumber' : 0,
                'CustomerId' : [self.CustomerId],
                'Surname' : [self.Surname],
                'CreditScore' : [self.Creditscore],
                'Geography' : [self.Geography],
                'Gender' : [self.Gender],
                'Age' : [self.Age],
                'Tenure' : [self.Tenure],
                'Balance' : [self.Balance],
                'NumOfProducts' : [self.NumOfProducts],
                'HasCrCard' : [self.HasCrCard],
                'IsActiveMember' : [self.IsActiveMember],
                'EstimatedSalary' : [self.EstimatedSalary],
        }
        return pd.DataFrame.from_dict(data)
    
class Predict:
    def __init__(self) -> None:
        self._load_model()

    def _load_model(self):
        train_model = TrainModel()
        self.model = train_model.load_model()

    def predict_churn(self,row: Customer):
        self.df = row.CustomerDataFrame()
        processor = Preprocess()
        self.df = processor.clean_df_predict(self.df)
        return self.df
    
print(os.getcwd())
predictor = Predict()

customer1 = [15647311,'Hill',608,'Spain','Female',41,1,83807.86,1,0,1,112542.58,0]
customer2 = [15619304,'Onio',502,'France','Female',42,8,159660.8,3,1,0,113931.57,1]
customer3 = [15701354,'Boni',699,'France','Female',39,1,0,2,0,0,93826.63,0]
customer4 = [15574012,'Chu',645,'Spain','Male',44,8,113755.78,2,1,0,149756.71,1]
customer5 = [15656148,'Obinna',376,'Germany','Female',29,4,115046.74,4,1,0,119346.88,1]

current_customer = customer3
churn_probability = predictor.predict_churn(Customer(current_customer))

train_model = TrainModel.load_model()
pred = train_model.predict_proba(churn_probability)[:, 1]
print(f'customer {current_customer[1]} is {pred}% likely to churn!')





