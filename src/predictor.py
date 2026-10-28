"""
Inference and Real-Time Prediction Engine.
Validates incoming customer attributes, applies the identical trained preprocessing
pipeline, computes calibrated churn probabilities, assigns risk tiers,
and generates profile-driven retention recommendations.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.config import CONFIG

logger = logging.getLogger(__name__)

class ChurnPredictor:
    """Provides single-record and batch inference using the persisted model pipeline."""

    DISCLAIMER = (
        "Note: Churn predictions are probabilistic statistical estimates based on historical "
        "telecommunication customer attrition patterns. They represent likelihoods, not deterministic guarantees."
    )

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or CONFIG.BEST_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at: {self.model_path}. Run training first.")
        
        logger.info(f"Loading trained churn pipeline from {self.model_path}")
        self.pipeline = joblib.load(self.model_path)

    def validate_customer_payload(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Validates customer attributes against the expected model feature schema."""
        missing = [col for col in CONFIG.NUMERICAL_COLS if col not in data]
        if missing:
            raise ValueError(f"Missing mandatory numerical features: {missing}")

        missing_cat = [col for col in CONFIG.CATEGORICAL_COLS if col not in data]
        if missing_cat:
            raise ValueError(f"Missing mandatory categorical features: {missing_cat}")

        # Construct single-row DataFrame
        df = pd.DataFrame([data])

        # Validate numerical ranges
        try:
            df["tenure"] = pd.to_numeric(df["tenure"], errors="raise").astype(float)
            if df["tenure"].iloc[0] < 0:
                raise ValueError("Tenure cannot be negative.")
        except Exception as e:
            raise ValueError(f"Invalid tenure value: {e}")

        try:
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="raise").astype(float)
            if df["MonthlyCharges"].iloc[0] < 0:
                raise ValueError("MonthlyCharges cannot be negative.")
        except Exception as e:
            raise ValueError(f"Invalid MonthlyCharges value: {e}")

        try:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            if pd.isna(df["TotalCharges"].iloc[0]):
                # Sensible imputation if tenure is 0 or omitted
                df["TotalCharges"] = df["MonthlyCharges"] * max(1.0, df["tenure"].iloc[0])
        except Exception as e:
            raise ValueError(f"Invalid TotalCharges value: {e}")

        # Ensure SeniorCitizen is mapped
        sc = str(data.get("SeniorCitizen", "0"))
        df["SeniorCitizen"] = "Yes" if sc in ("1", "Yes", "true", "True") else "No"

        # Reorder to match exact training column list
        ordered_cols = CONFIG.NUMERICAL_COLS + CONFIG.CATEGORICAL_COLS
        return df[ordered_cols]

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes inference for an individual customer profile.
        
        Returns:
            Dict containing predicted class, probability, risk tier,
            and prescriptive retention recommendations.
        """
        df_input = self.validate_customer_payload(customer_data)

        # Predict probability
        if hasattr(self.pipeline, "predict_proba"):
            prob = float(self.pipeline.predict_proba(df_input)[0, 1])
        else:
            prob = float(self.pipeline.predict(df_input)[0])

        pred_class = int(prob >= 0.50)
        prob_pct = round(prob * 100, 2)

        # Risk tiering
        if prob >= 0.60:
            risk_tier = "High Risk"
            badge_color = "red"
        elif prob >= 0.35:
            risk_tier = "Moderate Risk"
            badge_color = "orange"
        else:
            risk_tier = "Low Risk"
            badge_color = "green"

        # Formulate actionable retention actions
        recommendations = self._generate_recommendations(customer_data, prob)

        return {
            "prediction": "Churn" if pred_class == 1 else "No Churn",
            "prediction_class": pred_class,
            "churn_probability": round(prob, 4),
            "churn_probability_pct": prob_pct,
            "risk_tier": risk_tier,
            "badge_color": badge_color,
            "recommendations": recommendations,
            "model_version": CONFIG.MODEL_VERSION,
            "disclaimer": self.DISCLAIMER
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs batch inference on an uploaded DataFrame."""
        out = df.copy()

        # Handle TotalCharges blanks if present
        if "TotalCharges" in out.columns:
            out["TotalCharges"] = pd.to_numeric(out["TotalCharges"].astype(str).str.strip(), errors="coerce")
            out["TotalCharges"] = out["TotalCharges"].fillna(0.0)

        if "SeniorCitizen" in out.columns:
            out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes", "0": "No", "1": "Yes"}).fillna("No")

        feature_cols = [c for c in (CONFIG.NUMERICAL_COLS + CONFIG.CATEGORICAL_COLS) if c in out.columns]
        if len(feature_cols) < len(CONFIG.NUMERICAL_COLS + CONFIG.CATEGORICAL_COLS):
            missing = set(CONFIG.NUMERICAL_COLS + CONFIG.CATEGORICAL_COLS) - set(feature_cols)
            raise ValueError(f"Uploaded batch is missing required features: {missing}")

        X_batch = out[CONFIG.NUMERICAL_COLS + CONFIG.CATEGORICAL_COLS]
        probs = self.pipeline.predict_proba(X_batch)[:, 1]
        preds = (probs >= 0.50).astype(int)

        out["churn_probability"] = np.round(probs, 4)
        out["predicted_churn"] = np.where(preds == 1, "Churn", "No Churn")
        out["risk_tier"] = np.where(probs >= 0.60, "High Risk", np.where(probs >= 0.35, "Moderate Risk", "Low Risk"))

        return out

    def _generate_recommendations(self, data: Dict[str, Any], prob: float) -> List[str]:
        """Provides prescriptive retention playbooks based on feature patterns."""
        recs = []
        contract = str(data.get("Contract", ""))
        internet = str(data.get("InternetService", ""))
        tech_support = str(data.get("TechSupport", ""))
        tenure = float(data.get("tenure", 0))
        payment = str(data.get("PaymentMethod", ""))

        if "Month-to-month" in contract:
            recs.append("Contract Strategy: Customer is on a flexible Month-to-month contract (primary churn driver). Offer a 15% discount incentive for switching to a 1-year or 2-year commitment.")

        if "Fiber optic" in internet and tech_support == "No":
            recs.append("Support Bundle: Fiber optic subscribers without dedicated Tech Support exhibit elevated attrition. Offer 3 months of complimentary Tech Support.")

        if "Electronic check" in payment:
            recs.append("Payment Conversion: Electronic check users encounter higher friction. Incentivize conversion to Automated Credit Card or Bank Auto-Pay with a $5 billing credit.")

        if tenure <= 6:
            recs.append("Onboarding Care: Customer is in the vulnerable first 6 months. Schedule a proactive check-in call from customer success.")

        if not recs:
            recs.append("Customer shows stable loyalty indicators. Continue standard engagement and quarterly satisfaction pulse surveys.")

        return recs

