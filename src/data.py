import pandas as pd
import numpy as np
import config
import os
# Import pipeline and preprocessing imputers and encoders
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load data
class LoadData:
    def _load_data(self):
        if not os.path.exists(config.DATA_PATH):
            raise FileExistsError(f'File not found!')
        self.df = pd.read_csv(config.DATA_PATH)

class Preprocess(LoadData):
    def __init__(self):
        super().__init__()

    def clean_df_predict(self,df=pd.DataFrame()):
        self.df = df
        self._drop_missing_values()
        self.df._encode_surnames(config.COLUMNS_TO_REMOVE)
        self._create_new_features()
        return self.df

    def clean_df(self):
        print(f'Data Cleaning Initiated')
        self._load_data()
        self._drop_missing_values()
        self.df._encode_surnames(config.COLUMNS_TO_REMOVE)
        self._create_new_features()
        self._save_csv()
        print(f'Data Cleaning Successful. File Location: {config.PROCESSED_DATA_PATH}')
        return self.df

    def _encode_surnames(self,columns: list[str]):
        encoder = LabelEncoder()
        self.df['Surname_encoded'] = encoder.fit_transform(self.df['Surname'])
        self.df.drop(columns, axis=1, in_place=True)
    
    def _drop_missing_values(self):
        self.df.dropna(axis=0, in_place=True)

    def _create_new_features(self):
        encoder = LabelEncoder()
        age = list(range(30,81,10))
        for i,ages in enumerate(age):
            if i == 0:
                condition = self.df['Age'] < ages
                self.df.loc[condition,'Generation'] = i*10
            elif i == len(age)-1:
                condition = self.df['Age'] > ages
                self.df.loc[condition,'Generation'] = i*10
            else:
                condition = (self.df['Age'] >= ages - 10) & (self.df['Age'] < ages)  
                self.df.loc[condition,'Generation'] = i*10

        # Combine geography and gender
        self.df["GeoGender"] = self.df['Geography'] + self.df['Gender']

        # Creates a binary column for customer balance
        self.df['HasBalance'] = 'N'
        self.df.loc[self.df['Balance'] > 0,'HasBalance'] = 'Y'

        # Interactions between Active members and number of products
        self.df['Active__Prod'] = self.df['NumOfProducts'] * self.df['IsActiveMember']

        # Number of product by generation
        self.df['Gen_Prod'] = self.df['Generation'] * self.df['NumOfProducts']

        # Number of product by generation
        self.df['Gen_Geo'] = encoder.fit_transform(self.df['Geography'] + self.df['Generation'].astype(str))

        return self.df
    
    def _save_csv(self):
        self.df.to_csv(config.PROCESSED_DATA_PATH, index = False)
# Data cleaning
# feature engineering
# get num and cat columns

