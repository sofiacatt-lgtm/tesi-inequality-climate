# conditions.py

from parameters import *
from functions import *

def Primacondizione():
    return EA_up()*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF < 0

def Secondacondizione():
    return EA_p_min()*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF > 0

def Terzacondizione():
    return EA_p_min()*(1+mu) - mu*(1+r) - 1/(1-alpha) < 0

def Quartacondizione():
    return EA_p_max()*(1+mu) - mu*(1+r) - 1/(1-alpha) > 0

# versioni shock e no-shock
def Primacondizionenoshock():
    return A_up_noshock()*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF < 0

def Secondacondizionenoshock():
    return A_p_noshock(0)*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF > 0

def Terzacondizionenoshock():
    return A_p_bar*(1+mu) - mu*(1+r) - 1/(1-alpha) < 0

def Quartacondizionenoshock():
    return A_p_bar*2*(1+mu) - mu*(1+r) - 1/(1-alpha) > 0
