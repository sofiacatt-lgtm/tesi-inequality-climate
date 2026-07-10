# simulations_endogenous.py
#
# Implementazione della soglia endogena
#
# Teoria implementata (riferimenti alle equazioni del file .tex):
#
#   x^F_{t+1} = x^F_t + f/(1-alpha) * (X_{t+1} - X_t)              
#   che per telescopia diventa 
#   x_t^F = x^F_0 + f/(1-alpha) * ( X_t - X_0 )
#
# Nessun'altra assunzione teorica e' stata aggiunta rispetto al testo:
# f (qui chiamato f_adapt per non confondersi con il nome pyth on
# builtin "f" e con f_L/f_R usati nel testo per i due rami della mappa)
# resta un parametro libero — il file .tex non ne fornisce una
# calibrazione numerica, quindi qui e' lasciato come input dell'utente.

import numpy as np
import parameters as _par
from functions import EA_p, EA_up
from dynamics import update_wealth
from simulations_shock import draw_shocks, A_p as _A_p_shock_draw, A_up as _A_up_shock_draw


# ═══════════════════════════════════════════════════════════════════════════
# 1. Legge di aggiornamento della soglia endogena (eq. threshold_moving)
# ═══════════════════════════════════════════════════════════════════════════

def endogenous_threshold(X_t, X_0, xF_0, f_adapt, alpha, clip_nonnegative=True):
    """
    x_t^F = x^F_0 + f_adapt/(1-alpha) * (X_t - X_0)

    Parametri
    ---------
    X_t, X_0 : ricchezza aggregata (attesa) al tempo t e al tempo 0,
               X_t = int x dD_t(x)  (nel codice: np.mean della ricchezza
               individuale, dato che la massa totale di agenti e' 1)
    xF_0     : soglia strutturale iniziale, F/(1+mu)
    f_adapt  : costante positiva di sensibilita' della soglia alla
               variazione di output aggregato (chiamata "f" nel .tex)
    alpha    : quota di consumo (stesso alpha del resto del modello)
    clip_nonnegative : se True (default), impone x_t^F >= 0. Questo NON
               e' richiesto esplicitamente dal testo -- e' un vincolo
               numerico che aggiungo per evitare soglie negative prive
               di senso economico quando X_t < X_0 di molto (es. f_adapt
               grande o fase iniziale fortemente contrattiva). Impostare
               False per riprodurre la formula esattamente come scritta
               nel .tex.
    """
    xF_t = xF_0 + (f_adapt / (1 - alpha)) * (X_t - X_0)
    if clip_nonnegative:
        xF_t = max(xF_t, 0.0)
    return xF_t


# ═══════════════════════════════════════════════════════════════════════════
# 2. Simulazione "expected" / deterministica (analoga a simulations_det.py)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_endogenous_det(initial_distribution, update_wealth_fn, T, f_adapt,
                             N=None, alpha=None, mu=None, r=None, Ybar=None,
                             A_p_bar=None, A_up_bar=None, F=None,
                             p=None, q=None, gamma_p=None, gamma_up=None,
                             xF_0=None, clip_nonnegative=True):
    """
    Analogo di simulate_det (simulations_det.py), ma con soglia x^F
    endogena secondo l'eq. threshold_moving. La produttivita' usata ad
    ogni passo e' quella attesa E[A_p], E[A_up] (funzioni EA_p, EA_up di
    functions.py), esattamente come in simulate_det.

    Ritorna
    -------
    history      : lista di array (N,), ricchezza individuale a t=0..T
    xF_history   : lista di soglie x^F_t, t=0..T
    n_p_history  : lista delle frazioni n^P_t, t=0..T
    X_history    : lista della ricchezza aggregata X_t = mean(x_t), t=0..T
    """
    if N        is None: N        = _par.N
    if alpha    is None: alpha    = _par.alpha
    if mu       is None: mu       = _par.mu
    if r        is None: r        = _par.r
    if Ybar     is None: Ybar     = _par.Ybar
    if A_p_bar  is None: A_p_bar  = _par.A_p_bar
    if A_up_bar is None: A_up_bar = _par.A_up_bar
    if F        is None: F        = _par.F
    if p        is None: p        = _par.p
    if q        is None: q        = _par.q
    if gamma_p  is None: gamma_p  = _par.gamma_p
    if gamma_up is None: gamma_up = _par.gamma_up

    eta_p  = p * gamma_p
    eta_up = q * gamma_up

    if xF_0 is None:
        xF_0 = F / (1 + mu)

    x = initial_distribution(N)
    X_0 = np.mean(x)

    history     = [x.copy()]
    xF_history  = [xF_0]
    n_p_history = [np.mean(x > xF_0)]
    X_history   = [X_0]

    for t in range(T):
        X_t  = np.mean(x)
        xF_t = endogenous_threshold(X_t, X_0, xF_0, f_adapt, alpha, clip_nonnegative)
        n_p  = np.mean(x > xF_t)

        Ap  = EA_p(A_p_bar, n_p, eta_p)
        Aup = EA_up(A_up_bar, eta_up)

        x = np.array([update_wealth_fn(xi, n_p, Ap, Aup,
                                        xF=xF_t, Ybar=Ybar, mu=mu, r=r, alpha=alpha)
                      for xi in x])

        history.append(x.copy())
        xF_history.append(xF_t)
        n_p_history.append(n_p)
        X_history.append(np.mean(x))

    return history, xF_history, n_p_history, X_history


# ═══════════════════════════════════════════════════════════════════════════
# 3. Simulazione con shock stocastici (analoga a simulations_shock.py)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_endogenous_shock(initial_distribution, update_wealth_fn, T, f_adapt,
                               N=None, alpha=None, mu=None, r=None, Ybar=None,
                               A_p_bar=None, A_up_bar=None, F=None,
                               p=None, q=None, gamma_p=None, gamma_up=None,
                               xF_0=None, clip_nonnegative=True):
    """
    Analogo di simulate_shock (simulations_shock.py), ma con soglia x^F
    endogena secondo l'eq. threshold_moving. Ad ogni periodo viene estratta
    una realizzazione degli shock lambda_p, lambda_up (draw_shocks), esattamente
    come in simulate_shock.

    Ritorna
    -------
    history, xF_history, n_p_history, X_history : come in simulate_endogenous_det
    """
    if N        is None: N        = _par.N
    if alpha    is None: alpha    = _par.alpha
    if mu       is None: mu       = _par.mu
    if r        is None: r        = _par.r
    if Ybar     is None: Ybar     = _par.Ybar
    if A_p_bar  is None: A_p_bar  = _par.A_p_bar
    if A_up_bar is None: A_up_bar = _par.A_up_bar
    if F        is None: F        = _par.F
    if p        is None: p        = _par.p
    if q        is None: q        = _par.q
    if gamma_p  is None: gamma_p  = _par.gamma_p
    if gamma_up is None: gamma_up = _par.gamma_up

    if xF_0 is None:
        xF_0 = F / (1 + mu)

    x = initial_distribution(N)
    X_0 = np.mean(x)

    history     = [x.copy()]
    xF_history  = [xF_0]
    n_p_history = [np.mean(x > xF_0)]
    X_history   = [X_0]

    for t in range(T):
        X_t  = np.mean(x)
        xF_t = endogenous_threshold(X_t, X_0, xF_0, f_adapt, alpha, clip_nonnegative)
        n_p  = np.mean(x > xF_t)

        lambda_p, lambda_up = draw_shocks(p, q, gamma_p, gamma_up)
        Ap_t  = _A_p_shock_draw(A_p_bar, n_p, lambda_p)
        Aup_t = _A_up_shock_draw(A_up_bar, lambda_up)

        x = np.array([update_wealth_fn(xi, n_p, Ap_t, Aup_t,
                                        xF=xF_t, Ybar=Ybar, mu=mu, r=r, alpha=alpha)
                      for xi in x])

        history.append(x.copy())
        xF_history.append(xF_t)
        n_p_history.append(n_p)
        X_history.append(np.mean(x))

    return history, xF_history, n_p_history, X_history


# ═══════════════════════════════════════════════════════════════════════════
# 4. Mappa empirica n^P_{t+1} = H(n^P_t)
# ═══════════════════════════════════════════════════════════════════════════

def empirical_H_map(n_p_history):
    """
    A partire dalla traiettoria simulata n^P_t, restituisce le coppie
    (n^P_t, n^P_{t+1}) t=0..T-1: una stima empirica della mappa
    n^P_{t+1} = H(n^P_t) discussa nel .tex (subparagraph "Dynamics of n^P_t").
    Non e' una H analitica (il testo non ne fornisce la forma chiusa,
    si limita a definirla implicitamente) ma la sua traccia lungo la
    traiettoria simulata.
    """
    n_p_history = np.asarray(n_p_history)
    return n_p_history[:-1], n_p_history[1:]