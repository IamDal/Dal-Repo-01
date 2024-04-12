import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import config
# Import data visualization libraries
import matplotlib.pyplot as plt 
from data import Preprocess
import seaborn as sns 
import pickle
# Import Classifiers
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class  TrainModel:
    def __init__(self):
        self.df = self._read_df()

    def _read_df(self):
        try:
            if not os.path.exists(config.PROCESSED_DATA_PATH):
                raise FileExistsError(f'Clean File not found!')
        except Exception as e:
            print(e)
            pdo = Preprocess()
            pdo.clean_df()
        return pd.read_csv(config.PROCESSED_DATA_PATH)

    def _split_data(self):
        self.X = self.df.drop('Exited', axis=1)
        self.y = self.df['Exited']
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(self.X,self.y,test_size=.2,random_state=42)

    def _create_pipeline(self):
        self.num = self.X.select_dtypes(include=['int64', 'float64']).columns
        self.col = self.X.select_dtypes(include=['object']).columns

        # Preprocessing for numerical data: imputation and scaling
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())])

        # Preprocessing for categorical data: imputation and one-hot encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.num),
                ('cat', categorical_transformer, self.col)])
        return preprocessor

    def _train_model(self, preprocessor):
        def fit(model):
            name = type(model.named_steps['model']).__name__
            print(f'Fitting to {name}')
            model.fit(self.Xtrain, self.Ytrain)
            predictions = model.predict_proba(self.Xtest)[:, 1]
            auc_roc = roc_auc_score(self.Ytest, predictions)
            return {'model_name' : name, 'score' : auc_roc, 'model': model}

        # Set hyperparameters for XGBClassifier
        XGB = XGBClassifier(**{'n_estimators': 810, 'learning_rate': 0.07921079869615913, 'max_depth': 5,
                                    'min_child_weight': 8, 'gamma': 0.27423983829634263, 'random_state': 42, 'objective': 'binary:logistic',
                                    'eval_metric': 'auc', 'n_jobs': -1})

        # Set hyperparameters for CatBoostClassifier
        CATB = CatBoostClassifier(**{'iterations': 830, 'learning_rate': 0.08238714339235984, 'depth': 5,
                                        'l2_leaf_reg': 0.8106903985997884, 'random_state': 42, 'verbose': 0})

        # Set hyperparameters for LGBMClassifier
        LGBM = LGBMClassifier(**{'n_estimators': 960, 'learning_rate': 0.031725771326186744, 'max_depth': 8, 'min_child_samples': 8, 'force_row_wise': True,
                                    'subsample': 0.7458307885861184, 'num_leaves': 10,'colsample_bytree': 0.5111460378911089, 'random_state': 42})

        # Create pipelines
        XGB_best = Pipeline(steps=[('preprocessor', preprocessor), ('model', XGB)])
        CAT_best = Pipeline(steps=[('preprocessor', preprocessor), ('model', CATB)])
        LGBM_best = Pipeline(steps=[('preprocessor', preprocessor), ('model', LGBM)])
        
        XGB_results = fit(XGB_best)
        LGBM_results = fit(LGBM_best)
        CAT_results = fit(CAT_best)

        return [LGBM_results,XGB_results,CAT_results]

    def _train_voting_classifier(self,models):
        # Create a VotingClassifier with the three XGBoost models
        voting = VotingClassifier(estimators=[
            ('Model1', models[0]['model']),
            ('Model2', models[1]['model']),
            ('Model3', models[2]['model'])
        ], voting='soft', weights = [0.5, 0.3, 0.2], flatten_transform=True)

        voting.fit(self.Xtrain, self.Ytrain)

        predictions = voting.predict_proba(self.Xtest)[:, 1]
        predict = voting.predict(self.Xtest)

        auc_roc = roc_auc_score(self.Ytest, predictions)
        acuu = accuracy_score(self.Ytest, predict) 
        return {'model':voting, 'auc_roc_score':auc_roc, 'accuracy':acuu}

    def _save_model(self, voting):
        with open(config.SAVE_MODEL,'wb') as f:
            pickle.dump(voting, f)

    @staticmethod
    def load_model():
        with open('model.pkl','rb') as f:
            saved_model = pickle.load(f)

    def train(self):
        print(f'Model training Initiated')
        self._split_data()
        print(f'Data split @80% train & 20% validate')
        preprocessor = self._create_pipeline()
        results = self._train_model(preprocessor)
        voting_model, score, accuracy = self._train_voting_classifier(results)
        print(f'saving...')
        self._save_model(voting_model)
        print(f'model successfully saved at: {config.SAVE_MODEL}')