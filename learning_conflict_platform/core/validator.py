import pandas as pd
import numpy as np

def validate_dataset(df, target_col):
    if df.shape[0] < 30:
        return False, "Dataset too small for conflict analysis (minimum 30 samples required)."

    if target_col not in df.columns:
        return False, "Target column not found in dataset."

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return False, "Target column must be numeric (regression datasets only)."

    feature_df = df.drop(columns=[target_col])

    if feature_df.shape[1] == 0:
        return False, "No feature columns found."

    # Accept numeric + boolean
    non_numeric_cols = [
        col for col in feature_df.columns
        if not (
            pd.api.types.is_numeric_dtype(feature_df[col]) or
            pd.api.types.is_bool_dtype(feature_df[col])
        )
    ]

    if non_numeric_cols:
        return False, f"Non-numeric features detected: {non_numeric_cols}"

    if df[target_col].nunique() == 1:
        return False, "Target has zero variance. Learning conflicts cannot be computed."

    return True, "Dataset valid for learning conflict analysis."
