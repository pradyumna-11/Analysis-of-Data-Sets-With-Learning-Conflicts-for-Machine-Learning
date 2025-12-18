import pandas as pd
import matplotlib.pyplot as plt

def plot_conflict_distribution(df):
    plt.figure()
    plt.hist(df["total_conflict"], bins=50)
    plt.xlabel("Total Conflict Level")
    plt.ylabel("Number of Samples")
    plt.title("Distribution of Learning Conflict Levels")
    plt.show()


def plot_rmse_comparison(rmse_before, rmse_after):
    plt.figure()
    plt.bar(["Before Cleaning", "After Cleaning"], [rmse_before, rmse_after])
    plt.ylabel("RMSE")
    plt.title("Model Performance Comparison")
    plt.show()


def plot_removed_samples(df, removed_count=20):
    df_sorted = df.sort_values("total_conflict", ascending=False)

    plt.figure()
    plt.plot(df_sorted["total_conflict"].values, label="All Samples")
    plt.axvline(x=removed_count, linestyle="--", label="Removed Threshold")
    plt.xlabel("Samples (sorted)")
    plt.ylabel("Total Conflict")
    plt.title("Removed High-Conflict Samples")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/conflict_scored_dataset.csv")

    plot_conflict_distribution(df)
    plot_rmse_comparison(0.1200, 0.0744)
    plot_removed_samples(df)
