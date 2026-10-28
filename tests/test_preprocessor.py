"""
Unit tests for leakage-free ColumnTransformer preprocessing.
"""

import pandas as pd
from src.preprocessor import build_preprocessor, get_transformed_feature_names

def test_preprocessor_fit_transform():
    preprocessor = build_preprocessor()
    sample_data = pd.DataFrame([{
        "tenure": 12,
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }])

    transformed = preprocessor.fit_transform(sample_data)
    feature_names = get_transformed_feature_names(preprocessor)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == len(feature_names)
    assert "tenure" in feature_names

def test_preprocessor_handles_unseen_categories():
    preprocessor = build_preprocessor()
    train_data = pd.DataFrame([{
        "tenure": 5, "MonthlyCharges": 30.0, "TotalCharges": 150.0,
        "gender": "Male", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "No", "PaymentMethod": "Mailed check"
    }])
    preprocessor.fit(train_data)

    test_unseen = train_data.copy()
    test_unseen["InternetService"] = "Satellite_HyperSpeed"  # Unseen category

    # handle_unknown='ignore' should silently encode unseen categories without throwing error
    transformed = preprocessor.transform(test_unseen)
    assert transformed.shape[0] == 1

