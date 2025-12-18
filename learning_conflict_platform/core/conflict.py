import numpy as np

def delta(z1, z2):
    return np.sqrt(np.mean((z1 - z2) ** 2))

def conflict_score(z1, z2, t1, t2, sigma=0.02):
    d = delta(z1, z2)
    weight = np.exp(-(d**2) / (2 * sigma**2))
    return abs(t1 - t2) * weight
