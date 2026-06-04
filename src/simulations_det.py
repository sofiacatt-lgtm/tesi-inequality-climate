# simulation_det.py

import numpy as np
from parameters import *
from functions import *
from dynamics import *
from conditions import *

def simulate_det(initial_distribution, update_wealth, T):
    """
    initial_distribution: funzione che genera N valori iniziali e li mette
    nell'array x
    """

    x = initial_distribution(N)
    history = [x.copy()]

    for t in range(T):
        n_p = np.mean(x > xF)
        #print('n_p=', n_p)
        A_p= EA_p(n_p)
        A_up= EA_up()
        '''aggiorno ricchezza'''
        x = np.array([update_wealth(xi, n_p, A_p, A_up) for xi in x])
        history.append(x.copy())

    return history