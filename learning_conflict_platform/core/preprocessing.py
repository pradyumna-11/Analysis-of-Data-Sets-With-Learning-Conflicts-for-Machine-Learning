import pandas as pd

def handle_missing_values(df, target_col, strategy="mean"):
    """
    Handle missing values in dataset.
    - Numeric features: mean / median
    - Categorical features: mode
    - Target NaNs: rows dropped
    """
    df = df.copy()

    feature_cols = [c for c in df.columns if c != target_col]

    # Numeric features
    numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mean())

    # Categorical features
    categorical_cols = df[feature_cols].select_dtypes(exclude=["number", "bool"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Target NaNs → drop rows
    if df[target_col].isnull().any():
        df = df.dropna(subset=[target_col])

    return df


def encode_categorical_features(df, target_col):
    """
    Encode categorical features safely for distance-based learning.
    """
    df = df.copy()

    feature_cols = [c for c in df.columns if c != target_col]
    categorical_cols = df[feature_cols].select_dtypes(exclude=["number", "bool"]).columns

    # Binary yes/no encoding
    for col in categorical_cols:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals <= {"yes", "no"}:
            df[col] = df[col].map({"yes": 1, "no": 0})

    # One-hot encode remaining categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Convert boolean columns to int (0/1)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df
