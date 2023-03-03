import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd

app = FastAPI()
app.title = "Servicio de Machine Learning Usando FastAPI"
app.version = "0.0.1"

class RegModel(BaseModel):
    tasa_crimen : float
    lz : float
    industrial : float
    rio : float
    nox : float
    cuartos : float
    edad : float
    distancia : float
    autopistas : float
    impuestos : float
    profesores : float
    status : float

model = load('modelo_entrenado.joblib')

@app.post("/")
async def predict(item : RegModel):
    #X = pd.json_normalize(item.__dict__)
    df = pd.DataFrame([item.dict().values()],columns = item.dict().keys())
    
    # df = pd.DataFrame([[key, item[key]] for key in item.keys()], columns=item.dict().keys())
    prediction = model.predict(df)
    return {"predict":int(prediction)}

