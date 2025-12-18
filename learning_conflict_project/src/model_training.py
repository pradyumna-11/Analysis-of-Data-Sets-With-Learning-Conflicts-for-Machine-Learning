import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_evaluate(df, feature_cols, target_col):
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return rmse


if __name__ == "__main__":
    df = pd.read_csv("data/conflict_scored_dataset.csv")

    # RMSE BEFORE cleaning
    rmse_before = train_and_evaluate(
        df,
        feature_cols=["x1", "x2"],
        target_col="y"
    )

    # Remove top conflicted samples
    df_sorted = df.sort_values(by="total_conflict", ascending=False)
    cleaned_df = df_sorted.iloc[20:]  # remove top 20 conflicts

    # RMSE AFTER cleaning
    rmse_after = train_and_evaluate(
        cleaned_df,
        feature_cols=["x1", "x2"],
        target_col="y"
    )

    print(f"RMSE before conflict removal: {rmse_before:.4f}")
    print(f"RMSE after conflict removal: {rmse_after:.4f}")
