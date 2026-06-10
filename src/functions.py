# functions.py

from parameters import *

def EA_p_min(A_p_bar):
    return A_p_bar * (1 - eta_p)

def EA_p_max(A_p_bar):
    return A_p_bar * (1 - eta_p) * 2

def EA_p(A_p_bar, n_p):
    return A_p_bar * (1 + n_p) * (1 - eta_p)

def EA_up(A_up_bar):
    return A_up_bar * (1 - eta_up)

def A_p_shock(A_p_bar, n_p):
    return A_p_bar * (1 - gamma_p) * (1 + n_p)

def A_up_shock(A_up_bar):
    return A_up_bar * (1 - gamma_up)

def A_p_noshock(A_p_bar, n_p):
    return A_p_bar * (1 + n_p)

def A_up_noshock(A_up_bar):
    return A_up_bar

def xp(mu, r, alpha, A_up_bar, Ybar):
    return (1-alpha)*Ybar / (1 - (1-alpha)*((1+mu)*EA_up(A_up_bar) - mu*(1+r)))

def xr(mu, r, alpha, A_p_bar, n_p, Ybar):
    return (1-alpha)*Ybar / (1 - (1-alpha)*((1+mu)*EA_p(A_p_bar, n_p) - mu*(1+r)))

def aR(mu, r, alpha, A_p_bar, n_p):
    return (1-alpha)*((1+mu)*EA_p(A_p_bar, n_p) - mu*(1+r))

def aL(mu, r, alpha, A_up_bar):
    return (1-alpha)*((1+mu)*EA_up(A_up_bar) - mu*(1+r))
