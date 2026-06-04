# functions.py

from parameters import *

def EA_p_min():
    return A_p_bar * (1 - eta_p)

def EA_p_max():
    return A_p_bar * (1 - eta_p) * 2

def EA_p(n_p):
    return A_p_bar * (1 + n_p) * (1 - eta_p)

def EA_up():
    return A_up_bar * (1 - eta_up)

def A_p_shock(n_p):
    return A_p_bar * (1 - gamma_p) * (1 + n_p)

def A_up_shock():
    return A_up_bar * (1 - gamma_up)

def A_p_noshock(n_p):
    return A_p_bar * (1 + n_p)

def A_up_noshock():
    return A_up_bar

def xp():
    return (1-alpha)*Ybar / (1 - (1-alpha)*((1+mu)*EA_up() - mu*(1+r)))

def xr(n_p):
    return (1-alpha)*Ybar / (1 - (1-alpha)*((1+mu)*EA_p(n_p) - mu*(1+r)))

def aR(n_p):
    return (1-alpha)*((1+mu)*EA_p(n_p) - mu*(1+r))

def aL():
    return (1-alpha)*((1+mu)*EA_up() - mu*(1+r))
