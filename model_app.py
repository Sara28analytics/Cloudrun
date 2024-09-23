from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

class Input(BaseModel):
    CONSOLE: object 
    YEAR: int
    CATEGORY: object 
    PUBLISHER: object 
    RATING: object 
    CRITICS_POINTS: float
    USER_POINTS: float

class Output(BaseModel):
    SalesInMillions: float
    status: str

#uvicorn model_app:app --reload 
@app.post("/predict")
def predict(data:Input) -> Output:
    #input
    X_input = pd.DataFrame([{'CONSOLE':  data.CONSOLE,'YEAR':  data.YEAR,'CATEGORY':  data.CATEGORY,'PUBLISHER':  data.PUBLISHER,'RATING':  data.RATING,'CRITICS_POINTS':  data.CRITICS_POINTS,'USER_POINTS':  data.USER_POINTS}])
    #loading the model
    model = joblib.load('vgamessales_pipeline_model.pkl')
    #predict using the model
    prediction = model.predict(X_input)

    #output
    return Output(SalesInMillions=prediction, status = "Hurray! You have made it")

#go to the terminal and run the below command
# uvicorn model_app:app --reload

                           
