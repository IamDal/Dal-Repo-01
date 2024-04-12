import os
import pandas as pd
import numpy as np

import config
from train import TrainModel
from data import Preprocess
import matplotlib.pyplot as plt 
import seaborn as sns 
import pickle

class Customer:
    def __init__(self, CustomerId, Surname, CreditScore, Geography,
       Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
       IsActiveMember, EstimatedSalary):
        self.CustomerId = CustomerId
        self.Surname = Surname
        self.Creditscore = CreditScore
        self.Geography = Geography
        self.Gender = Gender
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.NumOfProducts = NumOfProducts
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary

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
churn_probability = predictor.predict_churn(Customer(15634602,'Hargrave',619,'France','Female',42,2,0,1,1,1,101348.88))

train_model = TrainModel.load_model()
pred = train_model.predict(churn_probability)
print(pred)




