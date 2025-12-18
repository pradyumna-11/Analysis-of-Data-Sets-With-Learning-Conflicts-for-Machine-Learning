import numpy as np
from core.conflict import conflict_score

def compute_total_conflict(df, features, target):
    X = df[features].values
    y = df[target].values
    N = len(df)
    C = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i != j:
                C[i] += conflict_score(X[i], X[j], y[i], y[j])
        C[i] /= (N - 1)

    df["total_conflict"] = C
    return df
