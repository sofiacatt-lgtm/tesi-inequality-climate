# distributions.py
import numpy as np

def uniform_initial(N, low=0, high=3):
    return np.random.uniform(low, high, N)

def lognormal_initial(N, mean=0, sigma=1):
    return np.random.lognormal(mean, sigma, N)

def bimodal_initial(N):
    return np.concatenate([
        np.random.normal(0.5, 0.1, N//2),
        np.random.normal(2.0, 0.2, N//2)
    ])
