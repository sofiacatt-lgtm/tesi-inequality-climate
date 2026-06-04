# simulation_shock.py

import numpy as np
from parameters import *
from dynamics import update_wealth

def draw_shocks(p, q, gamma_p, gamma_up):
    lambda_p = gamma_p if np.random.rand() < p else 0.0
    lambda_up = gamma_up if np.random.rand() < q else 0.0
    return lambda_p, lambda_up

def A_p(n_p, lambda_p):
    return A_p_bar * (1 + n_p) * (1 - lambda_p)

def A_up(lambda_up):
    return A_up_bar * (1 - lambda_up)

def simulate_shock(initial_distribution, update_wealth, T):

    x = initial_distribution(N)
    history = [x.copy()]

    for t in range(T):
        n_p = np.mean(x > xF)
        lambda_p, lambda_up = draw_shocks(p, q, gamma_p, gamma_up)
        Ap_t = A_p(n_p, lambda_p)
        Aup_t = A_up(lambda_up)
        x = np.array([update_wealth(xi, n_p, Ap_t, Aup_t) for xi in x])
        history.append(x.copy())

    return history
