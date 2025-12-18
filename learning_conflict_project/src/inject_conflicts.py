import numpy as np
import pandas as pd

def inject_learning_conflicts(
    clean_data_path="data/clean_dataset.csv",
    n_conflicts=20
):
    # Load clean dataset
    df = pd.read_csv(clean_data_path)

    conflict_rows = df.sample(n=n_conflicts, random_state=42)

    conflicted_samples = []

    for _, row in conflict_rows.iterrows():
        x1 = row['x1']
        x2 = row['x2']

        # Create a conflicting target (random large value)
        wrong_y = np.random.uniform(df['y'].max(), df['y'].max() * 2)
        #wrong_y = row['y'] + np.random.uniform(0.05, 0.1)


        conflicted_samples.append({
            'x1': x1,
            'x2': x2,
            'y': wrong_y
        })

    conflict_df = pd.DataFrame(conflicted_samples)

    # Combine clean + conflicted data
    contaminated_df = pd.concat([df, conflict_df], ignore_index=True)

    contaminated_df.to_csv("data/contaminated_dataset.csv", index=False)

    print(f"✅ Injected {n_conflicts} learning conflicts successfully")

if __name__ == "__main__":
    inject_learning_conflicts()
