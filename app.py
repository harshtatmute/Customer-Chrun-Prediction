from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import uvicorn
import pandas as pd
import joblib

model = pickle.load(open("best random forest model.pkl", "rb"))
model_features = joblib.load(open("model_features.pkl","rb"))

app = FastAPI(title="Customer Churn Prediction API")

# We need to:
# Send raw input (19 fields) from user to API
# Convert it into 30-feature one-hot encoded format (just like in training)
# Predict using the model
# Return output: "Churn" or "No Churn"


class CustomerData(BaseModel):
    gender : str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Welcome to the Customer Churn Prediction API. Use /docs to test."}

@app.post("/predict")
def predict_churn(data : CustomerData):
   #Convert input to dictionary
   input_dict = data.model_dump()
   input_df = pd.DataFrame([input_dict])

    # One-hot encode and match feature order
   input_encoded = pd.get_dummies(input_df)
   for col in model_features:
       if col not in input_encoded.columns:
           input_encoded[col]=0
   input_encoded = input_encoded[model_features]

   prediction = model.predict(input_encoded)[0]
   result = "Churn" if prediction == 1 else "No Churn"

   return {"Prediction": result}

