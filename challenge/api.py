import fastapi
from model import DelayModel
from pydantic import BaseModel
import joblib 
import os
import pandas as pd

app = fastapi.FastAPI()
model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
trained_model = joblib.load(model_path)
model = DelayModel()

class PredictionRequest(BaseModel):
   Fecha-I: object
   Vlo-I: object
   Ori-I: object
   Des-I: object
   Emp-I: object
   Fecha-O: object
   Vlo-O: object
   Ori-O: object
   Des-O: object
   Emp-O: object
   DIA: int 
   MES: int 
   AÑO: int 
   DIANOM: object
   TIPOVUELO: object
   OPERA: object
   SIGLAORI: object
   SIGLADES: object


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: PredictionRequest) -> dict:
    df = pd.DataFrame(data.dict())
    features = model.preprocess
    predictions = trained_model.predict(features)
    return predictions


