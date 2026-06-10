# conditions.py

from parameters import *
from functions import *

def Primacondizione(mu, r, alpha, xF, A_up_bar, Ybar):
    """Diseguaglianza 35: in xF deve stare sotto la bisettrice da sx (expected)"""
    if (EA_up(A_up_bar)*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF <0):
        return True
    else :
        return False

def Secondacondizione(mu, r, alpha, xF, A_p_bar, Ybar):
    """Diseguaglianza 36: in xF deve stare sopra la bisettrice da dx(expected)"""
    if (EA_p_min(A_p_bar)*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF >0):
        return True
    else :
        return False

def Terzacondizione(mu, r, alpha, xF, A_p_bar, Ybar):
    """Diseguaglianza 37: slope dx < 1 almeno nel valore minimo(expected)"""
    if (EA_p_min(A_p_bar)*(1+mu) - mu*(1+r) - 1/(1-alpha) <0):
        return True
    else :
        return False

def Quartacondizione(mu, r, alpha, xF, A_p_bar, Ybar):
    """Diseguaglianza 47: slope dx >1 almeno nel valore massimo(expected)"""
    if (EA_p_max(A_p_bar)*(1+mu) - mu*(1+r) - 1/(1-alpha) >0):
        return True
    else :
        return False


#no shock

def Primacondizionenoshock(mu, r, alpha, xF, A_up_bar, Ybar):
    """Diseguaglianza 35 senza shock: in xF deve stare sotto la bisettrice da sx"""
    if (A_up_noshock(A_up_bar)*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF <0):
        return True
    else :
        return False

def Secondacondizionenoshock(mu, r, alpha, xF, A_p_bar, Ybar):
    """Diseguaglianza 36 senza shock: in xF deve stare sopra la bisettrice da dx nel valore minimo"""
    if (A_p_noshock(A_p_bar, 0)*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF >0):
        return True
    else :
        return False

def Terzacondizionenoshock(mu, r, alpha, xF, A_p_bar, Ybar):
    """Diseguaglianza 37 senza shock: slope dx < 1 almeno nel valore minimo"""
    if (A_p_noshock(A_p_bar, 0)*(1+mu) - mu*(1+r) - 1/(1-alpha) <0):
        return True
    else :
        return False

def Quartacondizionenoshock(mu, r, alpha, xF, A_p_bar, Ybar):
    """Diseguaglianza 47 senza shock: slope dx >1 almeno nel valore massimo"""
    if (A_p_noshock(A_p_bar, 1)*(1+mu) - mu*(1+r) - 1/(1-alpha) >0):
        return True
    else :
        return False

#shock
def Primacondizioneshock(mu, r, alpha, xF, A_up_bar, Ybar):
    """Diseguaglianza 35: in xF deve stare sotto la bisettrice da sx"""
    if (A_up_shock(A_up_bar)*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF <0):
        return True
    else :
        return False

def Secondacondizioneshock(mu, r, alpha, xF, A_p_bar, Ybar ):
    """Diseguaglianza 36: in xF deve stare sopra la bisettrice da dx nel valore minimo"""
    if (A_p_shock(A_p_bar, 0)*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF >0):
        return True
    else :
        return False

def Terzacondizioneshock(mu, r, alpha, xF, A_p_bar, Ybar):
    """Diseguaglianza 37: slope dx < 1 almeno nel valore minimo"""
    if (A_p_shock(A_p_bar, 0)*(1+mu) - mu*(1+r) - 1/(1-alpha) <0):
        return True
    else :
        return False

def Quartacondizioneshock(mu, r, alpha, xF, A_p_bar, Ybar):
    """Diseguaglianza 47: slope dx >1 almeno nel valore massimo"""
    if (A_p_shock(A_p_bar, 1)*(1+mu) - mu*(1+r) - 1/(1-alpha) >0):
        return True
    else :
        return False
