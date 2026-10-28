"""
Model Training, Cross-Validation, Selection, and Evaluation Engine.
Trains Baseline, Logistic Regression, Random Forest, XGBoost, and LightGBM models.
Conducts 5-fold stratified cross-validation on the training set to prevent data leakage,
selects the champion model, evaluates all models on the held-out test partition,
and persists artifacts with real computed metrics.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# Silence loky cpu detection warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import CONFIG
from src.data_loader import TelcoDataLoader
from src.preprocessor import build_preprocessor, get_transformed_feature_names

# Optional gradient boosted tree libraries
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def is_lightgbm_compatible() -> bool:
    """Verifies that LightGBM C-libraries are compatible with current Windows OS environment."""
    try:
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=2, verbose=-1)
        m.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1]))
        return True
    except Exception as e:
        logger.warning(f"LightGBM runtime check failed ({e}). Excluding LightGBM from candidate pool.")
        return False

HAS_LIGHTGBM = is_lightgbm_compatible()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Orchestrates model comparison, tuning, selection, and artifact persistence."""

    def __init__(self):
        self.preprocessor = build_preprocessor()

    def get_candidate_models(self) -> Dict[str, Any]:
        """Instantiates candidate classification algorithms with class imbalance handling."""
        candidates = {
            "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=CONFIG.RANDOM_STATE
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                class_weight="balanced_subsample",
                random_state=CONFIG.RANDOM_STATE,
                n_jobs=-1
            )
        }

        if HAS_XGBOOST:
            # Positive weight ~ 5174 / 1869 = 2.76
            candidates["XGBoost"] = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                scale_pos_weight=2.76,
                random_state=CONFIG.RANDOM_STATE,
                eval_metric="logloss"
            )

        if HAS_LIGHTGBM:
            candidates["LightGBM"] = LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                class_weight="balanced",
                random_state=CONFIG.RANDOM_STATE,
                verbose=-1
            )

        return candidates

    def run_training_and_evaluation(self) -> Dict[str, Any]:
        """Executes the full machine learning training and evaluation cycle."""
        logger.info("Initiating Telco Churn Machine Learning Training Workflow...")
        start_time = time.time()

        # 1. Load data and split
        X, y, df_clean = TelcoDataLoader.load_clean_data()
        X_train, X_test, y_train, y_test = TelcoDataLoader.get_train_test_split(X, y)

        candidates = self.get_candidate_models()
        cv = StratifiedKFold(n_splits=CONFIG.CV_FOLDS, shuffle=True, random_state=CONFIG.RANDOM_STATE)

        cv_results = {}
        model_pipelines = {}
        evaluation_results = {}

        best_model_name = None
        best_cv_score = -1.0

        # 2. Cross-validation strictly on training set
        logger.info(f"Conducting {CONFIG.CV_FOLDS}-fold stratified CV across {len(candidates)} candidate models...")
        for name, model in candidates.items():
            pipe = Pipeline(steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", model)
            ])

            if name == "Baseline (Most Frequent)":
                scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
                mean_cv = float(np.mean(scores))
                std_cv = float(np.std(scores))
                cv_results[name] = {"metric": "Accuracy (Baseline)", "mean": round(mean_cv, 4), "std": round(std_cv, 4)}
            else:
                scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
                mean_cv = float(np.mean(scores))
                std_cv = float(np.std(scores))
                cv_results[name] = {"metric": "ROC-AUC", "mean": round(mean_cv, 4), "std": round(std_cv, 4)}
                
                # Model selection criterion (highest CV ROC-AUC on train partition)
                if mean_cv > best_cv_score:
                    best_cv_score = mean_cv
                    best_model_name = name

            # Fit pipeline on full training partition
            pipe.fit(X_train, y_train)
            model_pipelines[name] = pipe

            # 3. Evaluate on held-out test partition
            y_pred = pipe.predict(X_test)
            
            # Predict probabilities if supported
            if hasattr(pipe.named_steps["classifier"], "predict_proba"):
                y_prob = pipe.predict_proba(X_test)[:, 1]
                test_roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
                test_brier = round(float(brier_score_loss(y_test, y_prob)), 4)
                
                # Curve coordinates (sample 25 points for JSON)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                indices = np.linspace(0, len(fpr) - 1, 25, dtype=int)
                roc_curve_data = {"fpr": [round(float(x), 4) for x in fpr[indices]], "tpr": [round(float(x), 4) for x in tpr[indices]]}

                p_vals, r_vals, _ = precision_recall_curve(y_test, y_prob)
                p_indices = np.linspace(0, len(p_vals) - 1, 25, dtype=int)
                pr_curve_data = {"precision": [round(float(x), 4) for x in p_vals[p_indices]], "recall": [round(float(x), 4) for x in r_vals[p_indices]]}
            else:
                y_prob = np.zeros(len(y_test))
                test_roc_auc = 0.50
                test_brier = 0.0
                roc_curve_data = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}
                pr_curve_data = {"precision": [float(y_test.mean()), float(y_test.mean())], "recall": [1.0, 0.0]}

            cm = confusion_matrix(y_test, y_pred).tolist()
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            evaluation_results[name] = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "roc_auc": test_roc_auc,
                "brier_score": test_brier,
                "confusion_matrix": cm,
                "classification_report": report_dict,
                "roc_curve": roc_curve_data,
                "pr_curve": pr_curve_data
            }

            logger.info(
                f"Model [{name}] Test Metrics -> "
                f"Accuracy: {evaluation_results[name]['accuracy']:.4f}, "
                f"Precision: {evaluation_results[name]['precision']:.4f}, "
                f"Recall: {evaluation_results[name]['recall']:.4f}, "
                f"F1: {evaluation_results[name]['f1_score']:.4f}, "
                f"ROC-AUC: {evaluation_results[name]['roc_auc']:.4f}"
            )

        # 4. Extract Feature Importance from Champion Pipeline
        champion_pipeline = model_pipelines[best_model_name]
        fitted_preprocessor = champion_pipeline.named_steps["preprocessor"]
        fitted_classifier = champion_pipeline.named_steps["classifier"]
        feature_names = get_transformed_feature_names(fitted_preprocessor)

        feature_importances = []
        if hasattr(fitted_classifier, "feature_importances_"):
            importances = fitted_classifier.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            for idx in sorted_idx:
                feature_importances.append({
                    "feature": feature_names[idx],
                    "importance": round(float(importances[idx]), 4)
                })
        elif hasattr(fitted_classifier, "coef_"):
            coeffs = np.abs(fitted_classifier.coef_[0])
            sorted_idx = np.argsort(coeffs)[::-1]
            for idx in sorted_idx:
                feature_importances.append({
                    "feature": feature_names[idx],
                    "importance": round(float(coeffs[idx]), 4)
                })

        # 5. Save Artifacts
        joblib.dump(champion_pipeline, CONFIG.BEST_MODEL_PATH)
        logger.info(f"Champion pipeline [{best_model_name}] persisted to {CONFIG.BEST_MODEL_PATH}")

        # Save test predictions for transparency and audit
        champion_prob = champion_pipeline.predict_proba(X_test)[:, 1] if hasattr(fitted_classifier, "predict_proba") else np.zeros(len(y_test))
        champion_pred = champion_pipeline.predict(X_test)
        test_audit_df = X_test.copy()
        test_audit_df["actual_churn"] = y_test.values
        test_audit_df["predicted_churn"] = champion_pred
        test_audit_df["churn_probability"] = champion_prob.round(4)
        test_audit_df.to_csv(CONFIG.TEST_PREDICTIONS_PATH, index=False)

        # Metrics manifest
        total_time = round(time.time() - start_time, 2)
        metrics_manifest = {
            "model_version": CONFIG.MODEL_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_duration_seconds": total_time,
            "dataset_source": CONFIG.DATASET_SOURCE,
            "dataset_rows": len(X),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "churn_class_ratio": round(float(y.mean()), 4),
            "champion_model": best_model_name,
            "cv_selection_metric": "Stratified 5-Fold ROC-AUC",
            "cv_results": cv_results,
            "test_evaluation": evaluation_results,
            "class_imbalance_note": (
                "The dataset has a ~73.5% non-churn vs ~26.5% churn distribution. "
                "Accuracy is deceptive because a dummy model predicts 73.5% accuracy. "
                "Evaluation prioritizes ROC-AUC, Recall, and F1-score to capture customers who actually churn."
            ),
            "explainability_disclaimer": (
                "Feature importance indicates statistical association with churn risk "
                "within the trained model, not causal ground truth."
            )
        }

        with open(CONFIG.METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_manifest, f, indent=2)

        with open(CONFIG.FEATURE_IMPORTANCE_PATH, "w", encoding="utf-8") as f:
            json.dump(feature_importances, f, indent=2)

        logger.info(f"ML Pipeline execution complete in {total_time}s. Champion: {best_model_name}")
        return metrics_manifest

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run_training_and_evaluation()
    print("\nTraining Finished Successfully!")
    print(f"Selected Champion Model: {results['champion_model']}")
    print(f"Test Accuracy: {results['test_evaluation'][results['champion_model']]['accuracy']}")
    print(f"Test Recall:   {results['test_evaluation'][results['champion_model']]['recall']}")
    print(f"Test ROC-AUC:  {results['test_evaluation'][results['champion_model']]['roc_auc']}")

