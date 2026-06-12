# dynamics.py

import parameters as _par

def update_wealth(x, n_p, A_p, A_up,
                  xF=None, Ybar=None, mu=None, r=None, alpha=None):
    """
    Aggiorna la ricchezza di un singolo agente.
    """
    if xF    is None: xF    = _par.F / (1 + _par.mu)
    if Ybar  is None: Ybar  = _par.Ybar
    if mu    is None: mu    = _par.mu
    if r     is None: r     = _par.r
    if alpha is None: alpha = _par.alpha

    if x > xF:
        w = Ybar + (x + mu*x)*A_p - mu*x*(1+r)
    else:
        w = Ybar + (x + mu*x)*A_up - mu*x*(1+r)
    return (1 - alpha) * w

def update_fitness_exp(U_old, payoff, delta=None):
    if delta is None: delta = _par.delta
    return (1 - delta)*U_old + delta*payoff
