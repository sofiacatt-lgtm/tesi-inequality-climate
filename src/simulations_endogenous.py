# simulations_endogenous.py
#
# Implementazione dello step "Endogenous adaptation investment"

# Teoria di base implementata (eq. "temporal dependancy" + "threshold_moving"
# del file .tex):
#
#   x^F_{t+1} = x^F_t + f (Y_t - Y_{t-1}) = x^F_t + f/(1-alpha)*(X_{t+1}-X_t)
#
# che per telescopia diventa la formula boxed
#
#   x_t^F = x^F_0 + f/(1-alpha) * (X_t - X_0)
#
# PROBLEMA DI NON-NEGATIVITA'
# ----------------------------
# La formula additiva sopra e' illimitata in X_t: se l'economia entra
# nel ramo povero e X_t scende abbastanza sotto X_0, x_t^F diventa
# negativo, il che non ha senso dato che x^F = F/(1+mu) e F>=0 e' un
# costo strutturale. Questo modulo offre DUE specificazioni formalmente
# valide per imporre x_t^F >= 0, selezionabili con threshold_mode:
#
#   'log'    (DEFAULT) - riparametrizzazione moltiplicativa/log-lineare
#            x_t^F = x^F_0 * exp( f/(1-alpha)*(X_t - X_0) )
#            Positiva per costruzione, in forma chiusa esatta (nessun
#            troncamento, nessuna dipendenza dal percorso). Coincide al
#            primo ordine con la formula lineare originale. Stessa idea
#            (in forma dei fattori di danno/output moltiplicativi) usata
#            nei modelli di integrated assessment (Nordhaus, DICE) per
#            garantire che il fattore di danno resti positivo per
#            costruzione — si veda la discussione nel README/tex.
#
#   'corner' (ALTERNATIVA) - soluzione d'angolo applicata correttamente
#            come RICORSIONE periodo-per-periodo:
#            x^F_{t+1} = max{0, x^F_t + f/(1-alpha)*(X_{t+1}-X_t)}
#            NOTA: questa e' diversa (e non equivalente) dal semplice
#            troncamento della formula chiusa x^F_0+f/(1-alpha)(X_t-X_0),
#            che "dimentica" le discese sotto zero avvenute a meta'
#            percorso se X_t poi risale. La versione ricorsiva corretta
#            e' quella implementata sotto in endogenous_threshold_corner_step
#            ed e' quella usata da simulate_endogenous_* con
#            threshold_mode='corner'. La versione chiusa-poi-troncata e'
#            lasciata come endogenous_threshold_naive_clip SOLO per
#            confronto/diagnostica (era quella, senza questo avvertimento,
#            implementata nella versione precedente di questo modulo).
#
import numpy as np
import parameters as _par
from functions import EA_p, EA_up
from dynamics import update_wealth
from simulations_shock import draw_shocks, A_p as _A_p_shock_draw, A_up as _A_up_shock_draw

THRESHOLD_MODES = ("log", "corner")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Leggi di aggiornamento della soglia endogena
# ═══════════════════════════════════════════════════════════════════════════

def endogenous_threshold_log(X_t, X_0, xF_0, f_adapt, alpha):
    """
    Specificazione moltiplicativa/log-lineare (DEFAULT):

        x_t^F = x^F_0 * exp( f_adapt/(1-alpha) * (X_t - X_0) )

    Sempre positiva per costruzione, forma chiusa esatta, nessuna
    dipendenza dal percorso di X:
        log x^F_{t+1} = log x^F_t + f_adapt/(1-alpha) * (X_{t+1}-X_t)
    che telescopizza esattamente come la derivazione originale.
    """
    return xF_0 * np.exp((f_adapt / (1 - alpha)) * (X_t - X_0))


def endogenous_threshold_corner_step(xF_prev, X_prev, X_curr, f_adapt, alpha):
    """
    Specificazione "soluzione d'angolo" (ALTERNATIVA), RICORSIONE
    periodo-per-periodo:

        x^F_{t+1} = max{0, x^F_t + f_adapt/(1-alpha) * (X_{t+1}-X_t)}

    Va chiamata in sequenza (stato xF_prev, X_prev portato avanti ad ogni
    step): si veda l'uso in simulate_endogenous_det/_shock con
    threshold_mode='corner'. Rispetta il vincolo x^F_t>=0 in ogni periodo
    passato, non solo alla fine.
    """
    return max(0.0, xF_prev + (f_adapt / (1 - alpha)) * (X_curr - X_prev))


def endogenous_threshold_naive_clip(X_t, X_0, xF_0, f_adapt, alpha):
    """
    NON RACCOMANDATA - tenuta solo per confronto/diagnostica.

    Applica il troncamento a zero una sola volta, al termine cumulato
    (X_t - X_0), invece che periodo per periodo. NON e' equivalente a
    endogenous_threshold_corner_step: se X_t scende abbastanza da
    "vorrebbe" rendere la soglia negativa e poi X risale, questa versione
    fa ripartire il calcolo da zero come se la discesa non fosse mai
    avvenuta, mentre la ricorsione corretta (corner_step) mantiene la
    soglia bloccata a zero fino a quando l'aumento di X non compensa
    esattamente la discesa registrata. Era quanto implementato (senza
    questo avvertimento) nella versione precedente di questo modulo.
    """
    return max(0.0, xF_0 + (f_adapt / (1 - alpha)) * (X_t - X_0))

#integrate for one step the threshold
def _update_threshold(mode, xF_prev, X_prev, X_curr, xF_0, X_0, f_adapt, alpha):
    if mode == "log":
        return endogenous_threshold_log(X_curr, X_0, xF_0, f_adapt, alpha)
    elif mode == "corner":
        return endogenous_threshold_corner_step(xF_prev, X_prev, X_curr, f_adapt, alpha)
    else:
        raise ValueError(f"threshold_mode deve essere uno di {THRESHOLD_MODES}, ricevuto '{mode}'")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Simulazione "expected" / deterministica (analoga a simulations_det.py)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_endogenous_det(initial_distribution, update_wealth_fn, T, f_adapt,
                             N=None, alpha=None, mu=None, r=None, Ybar=None,
                             A_p_bar=None, A_up_bar=None, F=None,
                             p=None, q=None, gamma_p=None, gamma_up=None,
                             xF_0=None, threshold_mode="log"):
    """
    Analogo di simulate_det, ma con soglia x^F endogena. La produttivita'
    usata ad ogni passo e' quella attesa E[A_p], E[A_up] (functions.py),
    esattamente come in simulate_det.

    threshold_mode : 'log' (default, sempre positiva, forma chiusa) o
                      'corner' (soluzione d'angolo, ricorsione con
                      troncamento a zero periodo-per-periodo).

    Ritorna
    -------
    history, xF_history, n_p_history, X_history : liste lunghe T+1
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
    if threshold_mode not in THRESHOLD_MODES:
        raise ValueError(f"threshold_mode deve essere uno di {THRESHOLD_MODES}")

    eta_p  = p * gamma_p
    eta_up = q * gamma_up

    if xF_0 is None:
        xF_0 = F / (1 + mu)

    x = initial_distribution(N)
    X_0 = np.mean(x)
    xF_0_val = xF_0

    history     = [x.copy()]
    xF_history  = [xF_0_val]
    n_p_history = [np.mean(x > xF_0_val)]
    X_history   = [X_0]

    for t in range(T):
        # stato al tempo t e' gia' registrato in posizione t delle liste sopra;
        # lo uso per determinare la transizione t -> t+1
        Ap  = EA_p(A_p_bar, n_p_history[t], eta_p)
        Aup = EA_up(A_up_bar, eta_up)

        x = np.array([update_wealth_fn(xi, n_p_history[t], Ap, Aup,
                                        xF=xF_history[t], Ybar=Ybar, mu=mu, r=r, alpha=alpha)
                      for xi in x])

        X_next  = np.mean(x)
        xF_next = _update_threshold(threshold_mode, xF_history[t], X_history[t], X_next,
                                     xF_0_val, X_0, f_adapt, alpha)
        n_p_next = np.mean(x > xF_next)

        history.append(x.copy())
        xF_history.append(xF_next)
        n_p_history.append(n_p_next)
        X_history.append(X_next)

    return history, xF_history, n_p_history, X_history


# ═══════════════════════════════════════════════════════════════════════════
# 3. Simulazione con shock stocastici (analoga a simulations_shock.py)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_endogenous_shock(initial_distribution, update_wealth_fn, T, f_adapt,
                               N=None, alpha=None, mu=None, r=None, Ybar=None,
                               A_p_bar=None, A_up_bar=None, F=None,
                               p=None, q=None, gamma_p=None, gamma_up=None,
                               xF_0=None, threshold_mode="log"):
    """
    Analogo di simulate_shock, ma con soglia x^F endogena. Ad ogni periodo
    viene estratta una realizzazione degli shock lambda_p, lambda_up
    (draw_shocks), esattamente come in simulate_shock.

    threshold_mode : vedi simulate_endogenous_det.
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
    if threshold_mode not in THRESHOLD_MODES:
        raise ValueError(f"threshold_mode deve essere uno di {THRESHOLD_MODES}")

    if xF_0 is None:
        xF_0 = F / (1 + mu)

    x = initial_distribution(N)
    X_0 = np.mean(x)
    xF_0_val = xF_0

    history     = [x.copy()]
    xF_history  = [xF_0_val]
    n_p_history = [np.mean(x > xF_0_val)]
    X_history   = [X_0]

    for t in range(T):
        lambda_p, lambda_up = draw_shocks(p, q, gamma_p, gamma_up)
        Ap_t  = _A_p_shock_draw(A_p_bar, n_p_history[t], lambda_p)
        Aup_t = _A_up_shock_draw(A_up_bar, lambda_up)

        x = np.array([update_wealth_fn(xi, n_p_history[t], Ap_t, Aup_t,
                                        xF=xF_history[t], Ybar=Ybar, mu=mu, r=r, alpha=alpha)
                      for xi in x])

        X_next  = np.mean(x)
        xF_next = _update_threshold(threshold_mode, xF_history[t], X_history[t], X_next,
                                     xF_0_val, X_0, f_adapt, alpha)
        n_p_next = np.mean(x > xF_next)

        history.append(x.copy())
        xF_history.append(xF_next)
        n_p_history.append(n_p_next)
        X_history.append(X_next)

    return history, xF_history, n_p_history, X_history


# ═══════════════════════════════════════════════════════════════════════════
# 4. Mappa empirica n^P_{t+1} = H(n^P_t)
# ═══════════════════════════════════════════════════════════════════════════

def empirical_H_map(n_p_history):
    """
    A partire dalla traiettoria simulata n^P_t, restituisce le coppie
    (n^P_t, n^P_{t+1}) t=0..T-1: una stima empirica della mappa
    n^P_{t+1} = H(n^P_t) discussa nel .tex (subparagraph "Dynamics of n^P_t").
    Non e' una H analitica ma
    la sua traccia lungo la traiettoria simulata.
    """
    n_p_history = np.asarray(n_p_history)
    return n_p_history[:-1], n_p_history[1:]