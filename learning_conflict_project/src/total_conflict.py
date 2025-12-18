import numpy as np
import pandas as pd
from conflict_level import compute_conflict_level

def compute_total_conflict(df, feature_cols, target_col, sigma=0.02):
    N = len(df)
    total_conflict = np.zeros(N)

    features = df[feature_cols].values
    targets = df[target_col].values

    for i in range(N):
        conflict_sum = 0.0
        for j in range(N):
            if i != j:
                c_ij = compute_conflict_level(
                    features[i],
                    features[j],
                    targets[i],
                    targets[j],
                    sigma
                )
                conflict_sum += c_ij

        total_conflict[i] = conflict_sum / (N - 1)

        if i % 100 == 0:
            print(f"Processed sample {i}/{N}")

    return total_conflict


if __name__ == "__main__":
    df = pd.read_csv("data/normalized_dataset.csv")

    df["total_conflict"] = compute_total_conflict(
        df,
        feature_cols=["x1", "x2"],
        target_col="y"
    )

    df.to_csv("data/conflict_scored_dataset.csv", index=False)
    print("✅ Total conflict scores computed")
