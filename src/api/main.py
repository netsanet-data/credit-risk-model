from fastapi import FastAPI
from src.api.pydantic_models import (
    CustomerData,
    PredictionResponse
)

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Credit Risk API Running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    customer: CustomerData
):

    risk_probability = (
        customer.Amount /
        (customer.Value + 1)
    )

    if risk_probability > 1:
        risk_probability = 1.0

    return PredictionResponse(
        risk_probability=risk_probability
    )