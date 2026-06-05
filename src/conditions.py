# conditions.py

from parameters import *
from functions import *

def Primacondizione():
    """Diseguaglianza 35: in xF deve stare sotto la bisettrice da sx"""
    if (EA_up()*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF <0):
        return True
    else :
        return False

def Secondacondizione():
    """Diseguaglianza 36: in xF deve stare sopra la bisettrice da dx"""
    if (EA_p_min()*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF >0):
        return True
    else :
        return False

def Terzacondizione():
    """Diseguaglianza 37: slope dx < 1 almeno nel valore minimo"""
    if (EA_p_min()*(1+mu) - mu*(1+r) - 1/(1-alpha) <0):
        return True
    else :
        return False

def Quartacondizione():
    """Diseguaglianza 47: slope dx >1 almeno nel valore massimo"""
    if (EA_p_max()*(1+mu) - mu*(1+r) - 1/(1-alpha) >0):
        return True
    else :
        return False


#funzioni utili con gli shock

def Primacondizionenoshock():
    """Diseguaglianza 35 senza shock: in xF deve stare sotto la bisettrice da sx"""
    if (A_up_noshock()*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF <0):
        return True
    else :
        return False

def Secondacondizionenoshock():
    """Diseguaglianza 36 senza shock: in xF deve stare sopra la bisettrice da dx nel valore minimo"""
    if (A_p_noshock(0)*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF >0):
        return True
    else :
        return False

def Terzacondizionenoshock():
    """Diseguaglianza 37 senza shock: slope dx < 1 almeno nel valore minimo"""
    if (A_p_bar*(1+mu) - mu*(1+r) - 1/(1-alpha) <0):
        return True
    else :
        return False

def Quartacondizionenoshock():
    """Diseguaglianza 47 senza shock: slope dx >1 almeno nel valore massimo"""
    if (A_p_bar*2*(1+mu) - mu*(1+r) - 1/(1-alpha) >0):
        return True
    else :
        return False

def Primacondizioneshock():
    """Diseguaglianza 35: in xF deve stare sotto la bisettrice da sx"""
    if (A_up_shock()*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF <0):
        return True
    else :
        return False

def Secondacondizioneshock():
    """Diseguaglianza 36: in xF deve stare sopra la bisettrice da dx nel valore minimo"""
    if (A_p_shock(0)*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF >0):
        return True
    else :
        return False

def Terzacondizioneshock():
    """Diseguaglianza 37: slope dx < 1 almeno nel valore minimo"""
    if (A_p_shock(0)*(1+mu) - mu*(1+r) - 1/(1-alpha) <0):
        return True
    else :
        return False

def Quartacondizioneshock():
    """Diseguaglianza 47: slope dx >1 almeno nel valore massimo"""
    if (A_p_shock(1)*(1+mu) - mu*(1+r) - 1/(1-alpha) >0):
        return True
    else :
        return False
