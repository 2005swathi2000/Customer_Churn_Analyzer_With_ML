"""
Configuration and constants for Telecom Customer Churn Prediction and Analytics.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class ChurnConfig:
    # Dataset and Artifact Paths
    DATA_PATH: Path = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    BEST_MODEL_PATH: Path = ARTIFACTS_DIR / "best_model_pipeline.joblib"
    METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"
    FEATURE_IMPORTANCE_PATH: Path = ARTIFACTS_DIR / "feature_importance.json"
    TEST_PREDICTIONS_PATH: Path = ARTIFACTS_DIR / "test_predictions.csv"

    # Schema Definitions
    TARGET_COL: str = "Churn"
    ID_COL: str = "customerID"

    NUMERICAL_COLS: List[str] = field(default_factory=lambda: [
        "tenure", "MonthlyCharges", "TotalCharges"
    ])

    CATEGORICAL_COLS: List[str] = field(default_factory=lambda: [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod"
    ])

    # Modeling Parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.20
    CV_FOLDS: int = 5
    MODEL_VERSION: str = "v1.2.0"
    DATASET_SOURCE: str = "IBM Telco Customer Churn Open Dataset (7,043 rows, 21 attributes)"

CONFIG = ChurnConfig()
CONFIG.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

