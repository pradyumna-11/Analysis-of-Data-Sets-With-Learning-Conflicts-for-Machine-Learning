import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_rmse(df):
    X = df[["x1", "x2"]].values
    y = df["y"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


df = pd.read_csv("data/conflict_scored_dataset.csv")

# IQR outlier removal
Q1 = df["y"].quantile(0.25)
Q3 = df["y"].quantile(0.75)
IQR = Q3 - Q1

df_iqr = df[(df["y"] >= Q1 - 1.5 * IQR) & (df["y"] <= Q3 + 1.5 * IQR)]

# Conflict-based removal
df_conflict = df.sort_values("total_conflict", ascending=False).iloc[20:]

print("RMSE using IQR outlier removal:", train_rmse(df_iqr))
print("RMSE using learning conflict removal:", train_rmse(df_conflict))
