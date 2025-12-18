import matplotlib.pyplot as plt

def conflict_histogram(df):
    fig, ax = plt.subplots()
    ax.hist(df["total_conflict"], bins=40)
    ax.set_title("Conflict Distribution")
    return fig
