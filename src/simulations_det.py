# simulation_det.py

import numpy as np
import parameters as _par
from functions import EA_p, EA_up
from dynamics import update_wealth
from conditions import *

def simulate_det(initial_distribution, update_wealth_fn, T,
                 N=None, alpha=None, mu=None, r=None, Ybar=None,
                 A_p_bar=None, A_up_bar=None, F=None,
                 p=None, q=None, gamma_p=None, gamma_up=None):
    """
    initial_distribution: funzione che genera N valori iniziali e li mette
    nell'array x
    """
    if N       is None: N       = _par.N
    if alpha   is None: alpha   = _par.alpha
    if mu      is None: mu      = _par.mu
    if r       is None: r       = _par.r
    if Ybar    is None: Ybar    = _par.Ybar
    if A_p_bar is None: A_p_bar = _par.A_p_bar
    if A_up_bar is None: A_up_bar = _par.A_up_bar
    if F       is None: F       = _par.F
    if p       is None: p       = _par.p
    if q       is None: q       = _par.q
    if gamma_p is None: gamma_p = _par.gamma_p
    if gamma_up is None: gamma_up = _par.gamma_up

    xF    = F / (1 + mu)
    eta_p  = p * gamma_p
    eta_up = q * gamma_up

    x = initial_distribution(N)
    history = [x.copy()]

    for t in range(T):
        n_p = np.mean(x > xF)
        Ap  = EA_p(A_p_bar, n_p, eta_p)
        Aup = EA_up(A_up_bar, eta_up)
        '''aggiorno ricchezza'''
        x = np.array([update_wealth_fn(xi, n_p, Ap, Aup,
                                       xF=xF, Ybar=Ybar, mu=mu, r=r, alpha=alpha)
                      for xi in x])
        history.append(x.copy())

    return history
