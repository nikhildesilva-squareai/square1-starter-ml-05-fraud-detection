"""
FastAPI scoring service:  uvicorn app:app --reload

Loads the trained model (fraud_model.joblib) and exposes POST /score, which
takes one transaction and returns a fraud probability + a flag at your chosen
threshold.

TODO:
  - load the model you saved in train.py
  - build the same features from the incoming transaction that you trained on
  - return {"fraud_probability": float, "is_fraud_flag": int}
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Square 1 — Fraud Detection System")

# Pick the operating threshold you justified from the precision/recall trade-off.
THRESHOLD = 0.5


class Transaction(BaseModel):
    amount: float
    hour_of_day: int
    country_code: str
    is_foreign: int
    account_age_days: int
    txns_last_1h: int
    is_new_device: int
    merchant_category: str | None = None


class ScoreResponse(BaseModel):
    fraud_probability: float
    is_fraud_flag: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    """Score a single transaction for fraud risk."""
    raise NotImplementedError("Implement score: load model, build features, predict_proba")
