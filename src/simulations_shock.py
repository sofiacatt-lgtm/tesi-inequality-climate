# simulation_shock.py

import numpy as np
import parameters as _par
from dynamics import update_wealth

def draw_shocks(p, q, gamma_p, gamma_up):
    lambda_p = gamma_p if np.random.rand() < p else 0.0
    lambda_up = gamma_up if np.random.rand() < q else 0.0
    return lambda_p, lambda_up

def A_p(A_p_bar, n_p, lambda_p):
    return A_p_bar * (1 + n_p) * (1 - lambda_p)

def A_up(A_up_bar, lambda_up):
    return A_up_bar * (1 - lambda_up)

def simulate_shock(initial_distribution, update_wealth_fn, T,
                   N=None, alpha=None, mu=None, r=None, Ybar=None,
                   A_p_bar=None, A_up_bar=None, F=None,
                   p=None, q=None, gamma_p=None, gamma_up=None):

    if N        is None: N        = _par.N
    if alpha    is None: alpha    = _par.alpha
    if mu       is None: mu       = _par.mu
    if r        is None: r        = _par.r
    if Ybar     is None: Ybar     = _par.Ybar
    if A_p_bar  is None: A_p_bar  = _par.A_p_bar
    if A_up_bar is None: A_up_bar = _par.A_up_bar
    if F        is None: F        = _par.F
    if p        is None: p        = _par.p
    if q        is None: q        = _par.q
    if gamma_p  is None: gamma_p  = _par.gamma_p
    if gamma_up is None: gamma_up = _par.gamma_up

    xF = F / (1 + mu)

    x = initial_distribution(N)
    history = [x.copy()]

    for t in range(T):
        n_p = np.mean(x > xF)
        lambda_p, lambda_up = draw_shocks(p, q, gamma_p, gamma_up)
        Ap_t  = A_p(A_p_bar, n_p, lambda_p)
        Aup_t = A_up(A_up_bar, lambda_up)
        x = np.array([update_wealth_fn(xi, n_p, Ap_t, Aup_t,
                                       xF=xF, Ybar=Ybar, mu=mu, r=r, alpha=alpha)
                      for xi in x])
        history.append(x.copy())

    return history
