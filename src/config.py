DATA_PATH = './data/Churn_Modelling.csv'
PROCESSED_DATA_PATH = './data/cleaned_data.csv'
COLUMNS_TO_REMOVE = ['CustomerId','Surname']
NUMERICAL_COLUMNS = ['Gen_Prod', 'CreditScore', 'Balance', 'NumOfProducts', 'HasCrCard',
       'Tenure', 'IsActiveMember', 'Active__Prod', 'Age', 'EstimatedSalary',
       'Generation', 'Gen_Geo', 'Surname_encoded']
CATEGORICAL_COLUMNS = ['GeoGender', 'Geography', 'Gender', 'HasBalance']
SAVE_MODEL = './model/model.pkl'