import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def rmse(df, features, target):
    X = df[features]
    y = df[target]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # handles NaN values
        ("model", LinearRegression())
    ])

    pipeline.fit(Xtr, ytr)
    pred = pipeline.predict(Xte)

    return np.sqrt(mean_squared_error(yte, pred))
