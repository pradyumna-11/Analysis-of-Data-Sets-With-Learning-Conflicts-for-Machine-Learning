import numpy as np
import pandas as pd

def compute_input_difference(z_i, z_j):
    """
    Compute normalized input difference δ_ij
    """
    diff = z_i - z_j
    return np.sqrt(np.mean(diff ** 2))


def compute_target_difference(t_i, t_j):
    """
    Compute normalized target difference T_ij
    """
    return abs(t_i - t_j)


if __name__ == "__main__":
    # Load normalized dataset
    df = pd.read_csv("data/normalized_dataset.csv")

    # Select two samples for testing
    z_i = df.loc[0, ["x1", "x2"]].values
    z_j = df.loc[1, ["x1", "x2"]].values

    t_i = df.loc[0, "y"]
    t_j = df.loc[1, "y"]

    delta_ij = compute_input_difference(z_i, z_j)
    T_ij = compute_target_difference(t_i, t_j)

    print(f"δ_ij (Input Difference): {delta_ij:.4f}")
    print(f"T_ij (Target Difference): {T_ij:.4f}")
