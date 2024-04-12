from train import TrainModel
import sys
import config
#sys.path.append('/opt/conda/lib/python3.12/site-packages')

import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def home():
    return{'message':'API is working'}

if __name__ == '__main__':
    uvicorn.run(app)
    train_model = TrainModel()
    train_model.train()