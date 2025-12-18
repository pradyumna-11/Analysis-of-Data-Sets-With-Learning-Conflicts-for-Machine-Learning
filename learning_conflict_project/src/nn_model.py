import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_nn(df):
    X = df[["x1", "x2"]].values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


df = pd.read_csv("data/conflict_scored_dataset.csv")

rmse_before = train_nn(df)
rmse_after = train_nn(df.sort_values("total_conflict", ascending=False).iloc[20:])

print("NN RMSE before cleaning:", rmse_before)
print("NN RMSE after cleaning:", rmse_after)
