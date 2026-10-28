"""
Integration tests for model persistence, evaluation metrics, and ROC-AUC.
"""

import json
from src.config import CONFIG

def test_model_artifact_persistence():
    assert CONFIG.BEST_MODEL_PATH.exists(), "Trained model pipeline joblib does not exist."
    assert CONFIG.METRICS_PATH.exists(), "Evaluation metrics JSON does not exist."
    assert CONFIG.FEATURE_IMPORTANCE_PATH.exists(), "Feature importance JSON does not exist."
    assert CONFIG.TEST_PREDICTIONS_PATH.exists(), "Test predictions CSV does not exist."

def test_champion_evaluation_metrics():
    with open(CONFIG.METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    champion = metrics["champion_model"]
    test_res = metrics["test_evaluation"][champion]

    # Verify that metrics are realistic and meet minimum data science benchmarks
    assert test_res["roc_auc"] >= 0.80, f"Expected ROC-AUC >= 0.80, got {test_res['roc_auc']}"
    assert test_res["accuracy"] >= 0.70, f"Expected Accuracy >= 0.70, got {test_res['accuracy']}"
    assert test_res["recall"] >= 0.70, f"Expected Recall >= 0.70, got {test_res['recall']}"

