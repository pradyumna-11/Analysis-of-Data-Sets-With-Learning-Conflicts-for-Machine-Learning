def min_max_normalize(df, features, target):
    df = df.copy()
    for col in features + [target]:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df
