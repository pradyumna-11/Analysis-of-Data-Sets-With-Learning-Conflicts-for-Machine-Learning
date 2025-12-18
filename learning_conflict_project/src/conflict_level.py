import numpy as np
import pandas as pd
from similarity_metrics import compute_input_difference, compute_target_difference

def compute_conflict_level(z_i, z_j, t_i, t_j, sigma=0.02):
    delta_ij = compute_input_difference(z_i, z_j)
    T_ij = compute_target_difference(t_i, t_j)

    weight = np.exp(-(delta_ij ** 2) / (2 * sigma ** 2))
    c_ij = T_ij * weight

    return c_ij


if __name__ == "__main__":
    df = pd.read_csv("data/normalized_dataset.csv")

    z_i = df.loc[0, ["x1", "x2"]].values
    z_j = df.loc[1, ["x1", "x2"]].values

    t_i = df.loc[0, "y"]
    t_j = df.loc[1, "y"]

    c_ij = compute_conflict_level(z_i, z_j, t_i, t_j)

    print(f"Conflict Level c_ij: {c_ij:.6f}")
