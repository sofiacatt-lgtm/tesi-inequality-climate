# functions.py

import parameters as _par

def EA_p_min(A_p_bar, eta_p=None):
    if eta_p is None: eta_p = _par.p * _par.gamma_p
    return A_p_bar * (1 - eta_p)

def EA_p_max(A_p_bar, eta_p=None):
    if eta_p is None: eta_p = _par.p * _par.gamma_p
    return A_p_bar * (1 - eta_p) * 2

def EA_p(A_p_bar, n_p, eta_p=None):
    if eta_p is None: eta_p = _par.p * _par.gamma_p
    return A_p_bar * (1 + n_p) * (1 - eta_p)

def EA_up(A_up_bar, eta_up=None):
    if eta_up is None: eta_up = _par.q * _par.gamma_up
    return A_up_bar * (1 - eta_up)


def A_p_shock(A_p_bar, n_p, gamma_p=None):
    if gamma_p is None: gamma_p = _par.gamma_p
    return A_p_bar * (1 - gamma_p) * (1 + n_p)

def A_up_shock(A_up_bar, gamma_up=None):
    if gamma_up is None: gamma_up = _par.gamma_up
    return A_up_bar * (1 - gamma_up)

def A_p_noshock(A_p_bar, n_p):
    return A_p_bar * (1 + n_p)

def A_up_noshock(A_up_bar):
    return A_up_bar


def xp(mu=None, r=None, alpha=None, A_up_bar=None, Ybar=None, eta_up=None):
    if r    is None: r    = _par.r
    if Ybar  is None: Ybar  = _par.Ybar
    if mu    is None: mu    = _par.mu
    if r     is None: r     = _par.r
    if alpha is None: alpha = _par.alpha
    if A_up_bar is None: A_up_bar = _par.A_up_bar
    if eta_up is None: eta_up = _par.eta_up

    return (1-alpha)*Ybar / (1 - (1-alpha)*((1+mu)*EA_up(A_up_bar, eta_up) - mu*(1+r)))

def xr(n_p, mu=None, r=None, alpha=None, A_p_bar=None, Ybar=None, eta_p=None):
    if r    is None: r    = _par.r
    if Ybar  is None: Ybar  = _par.Ybar
    if mu    is None: mu    = _par.mu
    if r     is None: r     = _par.r
    if alpha is None: alpha = _par.alpha
    if A_p_bar is None: A_p_bar = _par.A_p_bar
    if eta_p is None: eta_p = _par.eta_p
    

    return (1-alpha)*Ybar / (1 - (1-alpha)*((1+mu)*EA_p(A_p_bar, n_p, eta_p) - mu*(1+r)))

def aR(n_p, mu=None, r=None, alpha=None, A_p_bar=None, eta_p=None):
    if r    is None: r    = _par.r
    if mu    is None: mu    = _par.mu
    if r     is None: r     = _par.r
    if alpha is None: alpha = _par.alpha
    if A_p_bar is None: A_p_bar = _par.A_p_bar
    if eta_p is None: eta_p = _par.eta_p
    

    return (1-alpha)*((1+mu)*EA_p(A_p_bar, n_p, eta_p) - mu*(1+r))

def aL(mu=None, r=None, alpha=None, A_up_bar=None, eta_up=None):
    if r    is None: r    = _par.r
    if mu    is None: mu    = _par.mu
    if r     is None: r     = _par.r
    if alpha is None: alpha = _par.alpha
    if A_up_bar is None: A_up_bar = _par.A_up_bar
    if eta_up is None: eta_up = _par.eta_up

    return (1-alpha)*((1+mu)*EA_up(A_up_bar, eta_up) - mu*(1+r))
