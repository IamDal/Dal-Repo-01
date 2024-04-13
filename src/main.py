from train import TrainModel
from predict import Predict, Customer
import sys
import config
sys.path.append('/opt/conda/lib/python3.12/site-packages')
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI

app = FastAPI()

class ChurnCustomer(BaseModel):
    CustomerId : int
    Surname : str
    CreditScore : int
    Geography : str
    Gender : str
    Age : float
    Tenure : int
    Balance : float
    NumOfProducts : int
    HasCrCard : int
    IsActiveMember : int
    EstimatedSalary : float

@app.get('/')
def home():
    return{'message':'API is working'}

@app.post('/predict')
def predict(customer: ChurnCustomer):
    new_customer = [customer.CustomerId,customer.Surname,customer.CreditScore,customer.Geography,
            customer.Gender,customer.Age,customer.Tenure,customer.Balance,customer.NumOfProducts,
            customer.HasCrCard,customer.IsActiveMember,customer.EstimatedSalary]
    
    customerA = Customer(new_customer)
    predictor = Predict()

    churn_probability = predictor.predict_churn(customerA)

    train_model = TrainModel.load_model()
    pred = train_model.predict_proba(churn_probability)[:, 1]
    return {'message':f'customer {customerA.Surname} is {pred}% likely to churn!'}


if __name__ == '__main__':
    train_model = TrainModel()
    train_model.train()
    uvicorn.run(app)