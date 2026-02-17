'''from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

pipeline = joblib.load("pipeline.pkl")

# -------- Input Schema --------
class IncomeInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# -------- API --------
@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: IncomeInput):

    df = pd.DataFrame([data.dict()])

    # rename columns to dataset names
    df.columns = [
        "age", "workclass", "fnlwgt", "education",
        "education.num", "marital.status", "occupation",
        "relationship", "race", "sex",
        "capital.gain", "capital.loss",
        "hours.per.week", "native.country"
    ]

    prediction = pipeline.predict(df)[0]

    return {"income_prediction": prediction}'''


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

pipeline = joblib.load("pipeline.pkl")

# ---------- Input Schema ----------
class IncomeInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# ---------- Output Schema ----------
class PredictionOutput(BaseModel):
    income_prediction: str

# ---------- Home Route ----------
@app.get("/")
def home():
    return {"message": "API Running"}

# ---------- Prediction Route ----------
@app.post("/predict", response_model=PredictionOutput)
def predict(data: IncomeInput):

    df = pd.DataFrame([data.dict()])

    df.columns = [
        "age", "workclass", "fnlwgt", "education",
        "education.num", "marital.status", "occupation",
        "relationship", "race", "sex",
        "capital.gain", "capital.loss",
        "hours.per.week", "native.country"
    ]

    prediction = pipeline.predict(df)[0]

    return {"income_prediction": prediction}
