"""
Data Loader and Schema Ingestion Module.
Loads the Telco Customer Churn dataset, validates feature contracts,
coerces inconsistent types (e.g. whitespace in TotalCharges), and prepares
leakage-free stratified train/test splits.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class TelcoDataLoader:
    """Manages loading, cleaning, and splitting of the Telco Churn dataset."""

    @classmethod
    def load_clean_data(cls, data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Loads raw CSV, enforces schema validation, handles missing TotalCharges,
        and extracts features X and target y.

        Returns:
            X (pd.DataFrame): Modeling features.
            y (pd.Series): Target vector (1 for Churn='Yes', 0 for Churn='No').
            df_clean (pd.DataFrame): Cleaned full dataset including customerID.
        """
        path = data_path or CONFIG.DATA_PATH
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found at: {path}")

        logger.info(f"Loading dataset from {path}")
        df = pd.read_csv(path)
        logger.info(f"Raw dataset shape: {df.shape}")

        # Validate required columns
        expected_cols = set([CONFIG.ID_COL, CONFIG.TARGET_COL] + CONFIG.NUMERICAL_COLS + CONFIG.CATEGORICAL_COLS)
        missing_cols = expected_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Schema violation! Missing required columns: {missing_cols}")

        df_clean = df.copy()

        # Handle TotalCharges: 11 rows have ' ' when tenure == 0
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # For new customers with tenure == 0, TotalCharges is legitimately 0.0
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

        # Standardize SeniorCitizen to string category ('No' / 'Yes') for consistent OneHotEncoding
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map({0: "No", 1: "Yes", "0": "No", "1": "Yes"}).fillna("No")

        # Encode target
        if CONFIG.TARGET_COL in df_clean.columns:
            df_clean[CONFIG.TARGET_COL] = df_clean[CONFIG.TARGET_COL].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
            y = df_clean[CONFIG.TARGET_COL].astype(int)
        else:
            y = None

        feature_cols = CONFIG.NUMERICAL_COLS + CONFIG.CATEGORICAL_COLS
        X = df_clean[feature_cols].copy()

        logger.info(f"Cleaned dataset ready: {len(X)} rows. Target class ratio: {y.mean():.4f}")
        return X, y, df_clean

    @classmethod
    def get_train_test_split(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = CONFIG.TEST_SIZE,
        random_state: int = CONFIG.RANDOM_STATE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Generates stratified train and test partitions to maintain class distribution.
        Prevents any preprocessing from leaking into test partition.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        logger.info(
            f"Stratified split created: Train={len(X_train)} (churn={y_train.mean():.2%}), "
            f"Test={len(X_test)} (churn={y_test.mean():.2%})"
        )
        return X_train, X_test, y_train, y_test

