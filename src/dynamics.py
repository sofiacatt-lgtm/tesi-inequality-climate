# dynamics.py

from parameters import *

def update_wealth(x, n_p, A_p, A_up):
    """
    Aggiorna la ricchezza di un singolo agente.
    """
    if x > xF:
        w = Ybar + (x + mu*x)*A_p - mu*x*(1+r)
    else:
        w = Ybar + (x + mu*x)*A_up - mu*x*(1+r)
    return (1 - alpha) * w

def update_fitness_exp(U_old, payoff):
    return (1 - delta)*U_old + delta*payoff
