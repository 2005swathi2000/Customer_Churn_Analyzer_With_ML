"""
Preprocessing and Feature Transformation Pipeline Module.
Implements scikit-learn ColumnTransformer to eliminate data leakage.
"""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import CONFIG

def build_preprocessor(
    numerical_cols: List[str] = CONFIG.NUMERICAL_COLS,
    categorical_cols: List[str] = CONFIG.CATEGORICAL_COLS
) -> ColumnTransformer:
    """
    Builds a reproducible scikit-learn ColumnTransformer.
    
    Numeric features:
        - Median Imputation
        - Z-score Standardization (StandardScaler)
        
    Categorical features:
        - Mode Imputation
        - One-Hot Encoding (handle_unknown='ignore' to gracefully handle unseen inference categories)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor

def get_transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts human-readable feature names from fitted ColumnTransformer."""
    feature_names = []
    
    # 1. Numeric features
    num_cols = preprocessor.transformers_[0][2]
    feature_names.extend(num_cols)

    # 2. Categorical features
    cat_pipe = preprocessor.transformers_[1][1]
    cat_cols = preprocessor.transformers_[1][2]
    encoder = cat_pipe.named_steps["encoder"]
    encoded_cats = encoder.get_feature_names_out(cat_cols)
    feature_names.extend(encoded_cats)

    return feature_names

