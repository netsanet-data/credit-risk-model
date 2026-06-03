from pydantic import BaseModel


class CustomerData(BaseModel):
    Amount: float
    Value: float


class PredictionResponse(BaseModel):
    risk_probability: float