"""
Unit and integration tests for real-time customer inference and input validation.
"""

import pytest
import pandas as pd
from src.predictor import ChurnPredictor

@pytest.fixture
def predictor():
    return ChurnPredictor()

@pytest.fixture
def sample_customer():
    return {
        "tenure": 2,
        "MonthlyCharges": 85.5,
        "TotalCharges": 171.0,
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }

def test_predict_single_high_risk(predictor, sample_customer):
    res = predictor.predict_single(sample_customer)

    assert "prediction" in res
    assert "churn_probability" in res
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert len(res["recommendations"]) > 0
    assert "Contract Strategy" in str(res["recommendations"])

def test_predict_single_validation_error(predictor, sample_customer):
    bad_customer = sample_customer.copy()
    bad_customer["tenure"] = -10  # Negative tenure

    with pytest.raises(ValueError, match="Tenure cannot be negative"):
        predictor.predict_single(bad_customer)

def test_predict_batch(predictor, sample_customer):
    df_batch = pd.DataFrame([sample_customer, sample_customer])
    out = predictor.predict_batch(df_batch)

    assert len(out) == 2
    assert "churn_probability" in out.columns
    assert "predicted_churn" in out.columns
    assert "risk_tier" in out.columns

