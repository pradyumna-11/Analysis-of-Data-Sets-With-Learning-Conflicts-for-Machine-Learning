import numpy as np
import pandas as pd

# Booth's function
def booths_function(x1, x2):
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

# Generate clean dataset
def generate_clean_data(n_samples=1000):
    x1 = np.random.uniform(-10, 10, n_samples)
    x2 = np.random.uniform(-10, 10, n_samples)

    y = booths_function(x1, x2)

    data = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'y': y
    })

    return data

if __name__ == "__main__":
    data = generate_clean_data()
    data.to_csv("data/clean_dataset.csv", index=False)
    print("✅ Clean dataset generated successfully")
