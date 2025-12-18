import pandas as pd

def min_max_normalize(df, feature_cols, target_col):
    df_norm = df.copy()

    # Normalize input features
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df_norm[col] = (df[col] - min_val) / (max_val - min_val)

    # Normalize target
    min_t = df[target_col].min()
    max_t = df[target_col].max()
    df_norm[target_col] = (df[target_col] - min_t) / (max_t - min_t)

    return df_norm

if __name__ == "__main__":
    df = pd.read_csv("data/contaminated_dataset.csv")

    normalized_df = min_max_normalize(
        df,
        feature_cols=["x1", "x2"],
        target_col="y"
    )

    normalized_df.to_csv("data/normalized_dataset.csv", index=False)
    print("✅ Data normalized successfully")
