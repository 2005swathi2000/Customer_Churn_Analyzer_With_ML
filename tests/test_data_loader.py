"""
Unit tests for data loading, TotalCharges cleaning, and stratified splitting.
"""

from src.data_loader import TelcoDataLoader
from src.config import CONFIG

def test_load_clean_data():
    X, y, df = TelcoDataLoader.load_clean_data()
    
    assert len(X) == 7043
    assert len(y) == 7043
    assert df["TotalCharges"].isna().sum() == 0
    # Ensure all required features are present
    for col in CONFIG.NUMERICAL_COLS:
        assert col in X.columns
    for col in CONFIG.CATEGORICAL_COLS:
        assert col in X.columns

def test_stratified_split_preserves_class_ratio():
    X, y, _ = TelcoDataLoader.load_clean_data()
    X_train, X_test, y_train, y_test = TelcoDataLoader.get_train_test_split(X, y)

    full_ratio = y.mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    # Ratios should match within 0.5%
    assert abs(train_ratio - full_ratio) < 0.005
    assert abs(test_ratio - full_ratio) < 0.005

