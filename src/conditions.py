# conditions.py

import numpy as np
import parameters as _par
from functions import (EA_p, EA_up, EA_p_min, EA_p_max,
                       A_p_noshock, A_up_noshock,
                       A_p_shock, A_up_shock,
                       aR, EA_p)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SELETTORE DI PRODUTTIVITÀ IN BASE AL MODE
# ═══════════════════════════════════════════════════════════════════════════════

# 'expected'  → analisi teorica con E[A_p]: non corrisponde a nessuna
#               realizzazione della simulazione, usato solo nella Sezione 1
# 'noshock'   → realizzazione effettiva senza shock (prob 1-p ad ogni step)
# 'shock'     → realizzazione effettiva con shock    (prob p   ad ogni step)

MODES            = ("expected", "noshock", "shock")
SIMULATION_MODES = ("noshock", "shock")   

def _get_productivities(mode, A_p_bar, A_up_bar, n_p_min=0, n_p_max=1,
                        p=None, gamma_p=None, gamma_up=None, q=None):
    """
    Restituisce (Ap_at_npmin, Ap_at_npmax, Aup) per il mode scelto.

    - Ap_at_npmin : produttività protetta quando n_p = 0 (lower bound slope destra)
    - Ap_at_npmax : produttività protetta quando n_p = 1 (upper bound slope destra)
    - Aup         : produttività non protetta (costante in n_p)

    Parametri override
    -------------------
    p, gamma_p, gamma_up, q : se forniti, sovrascrivono i valori di default
        letti da parameters.py. In mode 'expected' vengono usati per
        costruire eta_p = p*gamma_p e eta_up = q*gamma_up. In mode 'shock'
        gamma_p e gamma_up vengono passati direttamente alle funzioni
        A_p_shock/A_up_shock (p e q non entrano in questo mode).
    """
    if mode not in MODES:
        raise ValueError(f"mode deve essere uno di {MODES}, ricevuto '{mode}'")

    if mode == "expected":
        eta_p  = (p * gamma_p) if (p is not None and gamma_p is not None) else None
        q_eff  = q if q is not None else _par.q
        eta_up = (q_eff * gamma_up) if gamma_up is not None else None
        Ap_min = EA_p_min(A_p_bar, eta_p)               # A_p_bar*(1-eta_p)
        Ap_max = EA_p_max(A_p_bar, eta_p)                # A_p_bar*(1-eta_p)*2
        Aup    = EA_up(A_up_bar, eta_up)

    elif mode == "noshock":
        Ap_min = A_p_noshock(A_p_bar, n_p_min) # A_p_bar*(1+0)
        Ap_max = A_p_noshock(A_p_bar, n_p_max) # A_p_bar*(1+1)
        Aup    = A_up_noshock(A_up_bar)

    else:  # shock
        Ap_min = A_p_shock(A_p_bar, n_p_min, gamma_p)   # A_p_bar*(1-gamma_p)*(1+0)
        Ap_max = A_p_shock(A_p_bar, n_p_max, gamma_p)   # A_p_bar*(1-gamma_p)*(1+1)
        Aup    = A_up_shock(A_up_bar, gamma_up)

    return Ap_min, Ap_max, Aup


def _Ap_at_np(mode, A_p_bar, n_p, p=None, gamma_p=None):
    """Produttività protetta al valore corrente di n_p."""
    if mode == "expected":
        eta_p = (p * gamma_p) if (p is not None and gamma_p is not None) else None
        return EA_p(A_p_bar, n_p, eta_p)
    if mode == "noshock":
        return A_p_noshock(A_p_bar, n_p)
    return A_p_shock(A_p_bar, n_p, gamma_p)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CONDIZIONI FONDAMENTALI + FUNZIONE PER VERIFICARLE + COND PRIMITIVE
# ═══════════════════════════════════════════════════════════════════════════════
def _cond_1_fund(Ap_min, Aup):
    """(1)  E(A_p) > E(A_up), verificata al valore minimo n_p=0 (E(A_p) e'
    crescente in n_p, quindi se vale li' vale per ogni n_p in [0,1])."""
    return Ap_min > Aup

def _cond_2a_fund(Ap_min, r):
    """(2a) E(A_p) > 1+r per ogni n_p in [0,1], verificata al minimo n_p=0:
    le famiglie protette preferiscono investire tutto piuttosto che
    depositare, per ogni n_p ammissibile."""
    return Ap_min > 1 + r

def _cond_11_fund(Aup, r):
    """(11) E(A_up) > 1+r: le famiglie non protette preferiscono investire
    tutto piuttosto che depositare."""
    return Aup > 1 + r


def check_foundational_assumptions(A_p_bar, A_up_bar, r=None,
                                    p=None, gamma_p=None, gamma_up=None, q=None,
                                    verbose=True):
    """
    Controlla le condizioni (1), (2a), (11) del .tex — le assunzioni sotto cui
    la mappa piecewise-affine (e quindi dynamics.update_wealth) e' la
    rappresentazione corretta della scelta ottima delle famiglie. Vanno
    verificate ogni volta che si cambiano A_p_bar, A_up_bar, r, p, q, gamma_p,
    gamma_up .

    Nota: sono condizioni sulle produttivita' ATTESE (mode 'expected'), perche'
    la scelta di portafoglio della famiglia e' fatta ex-ante rispetto allo
    shock (si veda "Investment choices" nel .tex): vanno controllate con le
    produttivita' attese indipendentemente da quale mode (noshock/shock) si usi
    poi per SIMULARE la dinamica realizzata — la scelta d*=0 e' presa prima che
    lo shock si realizzi.

    Ritorna True se tutte e tre valgono; altrimenti, stampa quali falliscono.
    """
    if r is None: r = _par.r
    Ap_min, _, Aup = _get_productivities('expected', A_p_bar, A_up_bar,
                                          p=p, gamma_p=gamma_p, gamma_up=gamma_up, q=q)
    c1  = _cond_1_fund(Ap_min, Aup)
    c2a = _cond_2a_fund(Ap_min, r)
    c11 = _cond_11_fund(Aup, r)

    if verbose:
        print("╔══ ASSUNZIONI FONDAZIONALI (1),(2a),(11) ══╗")
        print(f"  E(A_p) al minimo (n_p=0) = {Ap_min:.4f}   E(A_up) = {Aup:.4f}   1+r = {1+r:.4f}")
        print(f"  (1)  E(Ap) > E(Aup) : {'OK' if c1 else 'VIOLATA'}")
        print(f"  (2a) E(Ap) > 1+r    : {'OK' if c2a else 'VIOLATA'}")
        print(f"  (11) E(Aup) > 1+r   : {'OK' if c11 else 'VIOLATA'}")
        if not (c1 and c2a and c11):
            print("  ATTENZIONE: con assunzioni violate, dynamics.update_wealth applica comunque")
            print("  la formula \"investi tutto\" incondizionatamente — la formula usata dal codice")
            print("  non corrisponde piu' alla scelta ottima della famiglia per questi parametri.")
        print()

    return c1 and c2a and c11


def _cond1(Aup, mu, r, alpha, xF, Ybar):
    """(5.2)  f_L(xF) < xF  →  poor branch contrattivo in xF."""
    return Aup*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF < 0

def _cond2(Ap, mu, r, alpha, xF, Ybar):
    """(6)  f_R(xF) > xF  valutata nel n_p che si mette dentro la chiamata di Ap."""
    return Ap*(1+mu) - mu*(1+r) - 1/(1-alpha) + Ybar/xF > 0
    
def _cond3(Ap_min, mu, r, alpha):
    """(7)    a_R < 1  quando n_p = 0  (lower bound slope destra)."""
    return Ap_min*(1+mu) - mu*(1+r) - 1/(1-alpha) < 0

def _cond4(Ap_max, mu, r, alpha):
    """(47)   a_R > 1  quando n_p = 1  (upper bound slope destra)."""
    return Ap_max*(1+mu) - mu*(1+r) - 1/(1-alpha) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SLOPE E N*
# ═══════════════════════════════════════════════════════════════════════════════

def slope_R(n_p, mode, A_p_bar, mu=None, r=None, alpha=None,
           p=None, gamma_p=None):
    """a_R(n_p) nel mode scelto."""
    if mu    is None: mu    = _par.mu
    if r     is None: r     = _par.r
    if alpha is None: alpha = _par.alpha
    Ap = _Ap_at_np(mode, A_p_bar, n_p, p=p, gamma_p=gamma_p)
    return (1-alpha)*((1+mu)*Ap - mu*(1+r))


def n_star(mode, A_p_bar, mu=None, r=None, alpha=None,
          p=None, gamma_p=None):
    """
    n^P critico tale che a_R(n*) = 1, nel mode scelto.

    A_p(n_p) = C*(1+n_p) con C che dipende dal mode:
      noshock  : C = A_p_bar
      shock    : C = A_p_bar*(1-gamma_p)
      expected : C = A_p_bar*(1-eta_p)   [solo per analisi teorica Sezione 1,
                                           non usare in contesto simulativo]

    Valutando A_p in n_p=0 si ottiene direttamente C.
    """
    if mode not in MODES:
        raise ValueError(f"mode deve essere uno di {MODES}, ricevuto '{mode}'")
    if mu    is None: mu    = _par.mu
    if r     is None: r     = _par.r
    if alpha is None: alpha = _par.alpha

    C = _Ap_at_np(mode, A_p_bar, 0, p=p, gamma_p=gamma_p)
    return (1/(1-alpha) + mu*(1+r)) / ((1+mu)*C) - 1


def n_star_simulation(A_p_bar, mu=None, r=None, alpha=None, gamma_p=None):
    """
    Restituisce i due n* rilevanti per la simulazione stocastica:
      - n*_noshock : soglia critica quando lo shock non si realizza
      - n*_shock   : soglia critica quando lo shock si realizza

    Non include 'expected' perché non corrisponde a nessuna
    realizzazione effettiva della produttività.

    Returns
    -------
    dict con chiavi 'noshock' e 'shock'
    """
    return {m: n_star(m, A_p_bar, mu=mu, r=r, alpha=alpha, gamma_p=gamma_p)
            for m in SIMULATION_MODES}


def n_dagger(mode, A_p_bar, mu=None, r=None, alpha=None, xF=None, Ybar=None,
             p=None, gamma_p=None):
    """
    n^P critico tale che f_R(xF) = xF  nel
    mode scelto.

    E' un secondo tipping point, DIVERSO da n_star: n_star riguarda la
    PENDENZA di f_R (a_R<1 o >1), n_dagger riguarda la POSIZIONE di f_R(xF)
    rispetto alla bisettrice. Sotto n_dagger, anche se il ramo destro fosse
    contrattivo (a_R<1), il suo punto fisso implicito cadrebbe fuori dal
    dominio ammissibile x>xF: nessuna famiglia puo' sostenere il regime
    ricco, e il regime e' 'unique_poor_attractor' invece di
    'poor_and_rich_attractor'.

    Vale sempre n_dagger <= n_star

        n_dagger = n_star - Ybar / ((1+mu) * xF * C)

    con C = A_p(n_p=0) nel mode scelto (stessa costante usata in n_star).
    Puo' essere negativo (fuori da [0,1]): in quel caso la condizione (6)
    vale gia' per ogni n_p ammissibile e la fase 'unique_poor_attractor' e'
    assente per costruzione (e' il caso dei parametri di default in
    parameters.py).
    """
    if mode not in MODES:
        raise ValueError(f"mode deve essere uno di {MODES}, ricevuto '{mode}'")
    if mu    is None: mu    = _par.mu
    if r     is None: r     = _par.r
    if alpha is None: alpha = _par.alpha
    if xF    is None: xF    = _par.xF
    if Ybar  is None: Ybar  = _par.Ybar

    C  = _Ap_at_np(mode, A_p_bar, 0, p=p, gamma_p=gamma_p)
    ns = n_star(mode, A_p_bar, mu=mu, r=r, alpha=alpha, p=p, gamma_p=gamma_p)
    return ns - Ybar / ((1 + mu) * xF * C)


def n_dagger_simulation(A_p_bar, mu=None, r=None, alpha=None, xF=None, Ybar=None, gamma_p=None):
    """Analogo di n_star_simulation, ma per n_dagger (vedi n_dagger())."""
    return {m: n_dagger(m, A_p_bar, mu=mu, r=r, alpha=alpha, xF=xF, Ybar=Ybar, gamma_p=gamma_p)
            for m in SIMULATION_MODES}


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TIPO DI RAMO 
# ═══════════════════════════════════════════════════════════════════════════════
def left_branch_type(mode, A_p_bar, A_up_bar,
                      mu=None, r=None, alpha=None, xF=None, Ybar=None,
                      p=None, gamma_p=None, gamma_up=None, q=None):
    """
    Caratterizza il ramo sinistro 

    Returns
    -------
    'poor_attractor' : f_L(xF)< xF
    'no_poor_attractor'   : f_L(xF)>xF
    """
    if mu    is None: mu    = _par.mu
    if r     is None: r     = _par.r
    if alpha is None: alpha = _par.alpha
    if xF    is None: xF    = _par.xF
    if Ybar  is None: Ybar  = _par.Ybar

    Ap_min, Ap_max, Aup = _get_productivities(mode, A_p_bar, A_up_bar,
                                             p=p, gamma_p=gamma_p, gamma_up=gamma_up, q=q)
    c1 = _cond1(Aup, mu, r, alpha, xF, Ybar)
    
    if c1 : return "poor_attractor"
    if not c1: return "no_poor_attractor"
    return "impossible"

def right_branch_type(mode, A_p_bar, A_up_bar,
                      mu=None, r=None, alpha=None, xF=None, Ybar=None,
                      p=None, gamma_p=None, gamma_up=None, q=None):
    """
    Caratterizza il ramo destro al variare di n_p in [0,1].

    Returns
    -------
    'always_contractive' : a_R < 1 per ogni n_p
    'phase_transition'   : esiste n* in (0,1)
    'always_expansive'   : a_R > 1 per ogni n_p
    """
    if mu    is None: mu    = _par.mu
    if r     is None: r     = _par.r
    if alpha is None: alpha = _par.alpha
    if xF    is None: xF    = _par.xF
    if Ybar  is None: Ybar  = _par.Ybar

    Ap_min, Ap_max, _ = _get_productivities(mode, A_p_bar, A_up_bar,
                                             p=p, gamma_p=gamma_p, gamma_up=gamma_up, q=q)
    c3 = _cond3(Ap_min, mu, r, alpha)
    c4 = _cond4(Ap_max, mu, r, alpha)

    if     c3 and not c4: return "always_contractive"
    if     c3 and     c4: return "phase_transition"
    if not c3 and     c4: return "always_expansive"
    return "impossible"


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  REGIME DATO n_p CORRENTE
# ═══════════════════════════════════════════════════════════════════════════════

def regime(n_p, mode, A_p_bar, A_up_bar,
          mu=None, r=None, alpha=None, xF=None, Ybar=None,
          p=None, gamma_p=None, gamma_up=None, q=None):
    if mu    is None: mu    = _par.mu
    if r     is None: r     = _par.r
    if alpha is None: alpha = _par.alpha
    if xF    is None: xF    = _par.xF
    if Ybar  is None: Ybar  = _par.Ybar

    Ap_min, Ap_max, Aup = _get_productivities(mode, A_p_bar, A_up_bar,
                                               p=p, gamma_p=gamma_p, gamma_up=gamma_up, q=q)

    Ap_np  = _Ap_at_np(mode, A_p_bar, n_p, p=p, gamma_p=gamma_p)   # produttività ESATTA a questo n_p

    c1     = _cond1(Aup,   mu, r, alpha, xF, Ybar)
    c2_np  = _cond2(Ap_np, mu, r, alpha, xF, Ybar)   # f_R(xF)>xF valutata ESATTAMENTE a n_p
    aR_lt1 = slope_R(n_p, mode, A_p_bar, mu, r, alpha, p=p, gamma_p=gamma_p) < 1
    btype  = right_branch_type(mode, A_p_bar, A_up_bar, mu, r, alpha, xF, Ybar,
                                p=p, gamma_p=gamma_p, gamma_up=gamma_up, q=q)

    if btype == "always_contractive":
        if not c1:        return "unique_rich_attractor"
        if c1 and c2_np:  return "poor_and_rich_attractor"
        return "unique_poor_attractor"

    if btype == "always_expansive":
        if not c1:            return "indefinite_growth"
        return "poor_attractor_and_explosion"

    if btype == "phase_transition":
        if not c1:
            return "unique_rich_attractor" if aR_lt1 else "indefinite_growth"
        if aR_lt1 and c2_np:     return "poor_and_rich_attractor"
        if aR_lt1 and not c2_np: return "unique_poor_attractor"
        return "poor_attractor_and_explosion"

    return "undetermined"


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAPPA DEL REGIME AL VARIARE DI n_p: array di stringhe in f di array di numeri
# ═══════════════════════════════════════════════════════════════════════════════

def regime_map(mode, A_p_bar, A_up_bar, n_grid=200, **kwargs):
    """Restituisce (n_p_values, regimes) per il mode scelto."""
    n_values = np.linspace(0, 1, n_grid)
    regimes  = [regime(n, mode, A_p_bar, A_up_bar, **kwargs) for n in n_values]
    return n_values, regimes


def regime_transitions(mode, A_p_bar, A_up_bar, n_grid=400, **kwargs):
    """
    Scandisce n_p in [0,1] con regime_map e restituisce tutti i punti in cui
    il regime cambia, SENZA assumere a priori quali o quante soglie esistano
    (rileva n_star, n_dagger, ed eventuali altre soglie non ancora derivate
    analiticamente). non dipende da
    dove pensiamo che siano i tipping point, li trova scandendo la griglia.

    Returns
    -------
    initial_regime : regime per n_p vicino a 0
    transitions    : lista di tuple (n_p_approx, regime_prima, regime_dopo),
                     in ordine di n_p crescente
    """
    n_values, regimes = regime_map(mode, A_p_bar, A_up_bar, n_grid=n_grid, **kwargs)
    transitions = []
    for i in range(1, len(n_values)):
        if regimes[i] != regimes[i - 1]:
            transitions.append((n_values[i], regimes[i - 1], regimes[i]))
    return regimes[0], transitions


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  DIAGNOSTICA
# ═══════════════════════════════════════════════════════════════════════════════

def _print_mode_block(mode, kw):
    """Stampa il blocco diagnostico per un singolo mode."""
    A_p_bar  = kw["A_p_bar"]
    A_up_bar = kw["A_up_bar"]
    mu, r, alpha = kw["mu"], kw["r"], kw["alpha"]
    xF, Ybar     = kw["xF"], kw["Ybar"]
    p, gamma_p   = kw.get("p"), kw.get("gamma_p")

    ns = n_star(mode, A_p_bar, mu=mu, r=r, alpha=alpha, p=p, gamma_p=gamma_p)
    nd = n_dagger(mode, A_p_bar, mu=mu, r=r, alpha=alpha, xF=xF, Ybar=Ybar, p=p, gamma_p=gamma_p)

    lbtype = left_branch_type(mode, **kw)
    rbtype = right_branch_type(mode, **kw)

    sep = "─" * 55
    print(sep)
    print(f" MODE: {mode.upper()}")
    print(sep)
    print(f"  Tipo ramo sinistro : {lbtype}")
    print(f"  Tipo ramo destro : {rbtype}")
    print(f"  n_dagger (soglia di POSIZIONE, f_R(xF)=xF) = {nd:.4f}  "
          + ("(in [0,1] )" if 0 < nd < 1 else "(not in [0,1] )"))
    print(f"  n*       (soglia di PENDENZA,  a_R=1)      = {ns:.4f}  "
          + ("(in [0,1] )" if 0 < ns < 1 else "(not in [0,1] )"))
    if nd > ns:
        print("  ATTENZIONE: n_dagger > n* — non dovrebbe succedere (n_dagger<=n* per costruzione); "
              "controllare i parametri.")
    print(f"  a_R(n_p = 0)     = {slope_R(0,               mode, A_p_bar, mu, r, alpha, p=p, gamma_p=gamma_p):.4f}")
    print(f"  a_R(n_p = n*)    = {slope_R(max(0,min(ns,1)),mode, A_p_bar, mu, r, alpha, p=p, gamma_p=gamma_p):.4f}")
    print(f"  a_R(n_p = 1)     = {slope_R(1,               mode, A_p_bar, mu, r, alpha, p=p, gamma_p=gamma_p):.4f}")

    # scansione automatica: non assume a priori quali/quante soglie esistano
    initial_regime, transitions = regime_transitions(mode, **kw)
    print(f"  Regime per n_p vicino a 0 : {initial_regime}")
    if not transitions:
        print("  Nessuna transizione di regime rilevata in [0,1].")
    for n_p_t, before, after in transitions:
        print(f"  transizione vicino a n_p≈{n_p_t:.4f} : {before}  →  {after}")
    print()


def print_regime_summary(A_p_bar=None, A_up_bar=None,
                         mu=None, r=None, alpha=None, xF=None, Ybar=None,
                         p=None, gamma_p=None, gamma_up=None, F=None, q=None):
    """
    Stampa la diagnostica separando l'analisi teorica (expected)
    dall'analisi simulativa (noshock e shock).

    Parametri liberamente controllabili
    ------------------------------------
    p         : probabilità di shock sul settore protetto (eta_p = p*gamma_p)
    q         : probabilità di shock sul settore non protetto (eta_up = q*gamma_up)
    gamma_p   : ampiezza dello shock sulla produttività protetta
    gamma_up  : ampiezza dello shock sulla produttività non protetta
    A_p_bar   : produttività di base del settore protetto
    A_up_bar  : produttività di base del settore non protetto
    F (o xF)  : soglia strutturale x_F. Se passi F, ha la stessa funzione di xF.
    """
    if A_p_bar  is None: A_p_bar  = _par.A_p_bar
    if A_up_bar is None: A_up_bar = _par.A_up_bar
    if mu       is None: mu       = _par.mu
    if r        is None: r        = _par.r
    if alpha    is None: alpha    = _par.alpha
    if xF       is None: xF       = F if F is not None else _par.xF
    if Ybar     is None: Ybar     = _par.Ybar
    if p        is None: p        = _par.p
    if q        is None: q        = _par.q
    if gamma_p  is None: gamma_p  = _par.gamma_p
    if gamma_up is None: gamma_up = _par.gamma_up

    kw = dict(A_p_bar=A_p_bar, A_up_bar=A_up_bar,
              mu=mu, r=r, alpha=alpha, xF=xF, Ybar=Ybar,
              p=p, gamma_p=gamma_p, gamma_up=gamma_up, q=q)

    print("╔══ PARAMETRI CORRENTI ══╗")
    print(f"  p = {p}   q = {q}   gamma_p = {gamma_p}   gamma_up = {gamma_up}")
    print(f"  A_p_bar = {A_p_bar}   A_up_bar = {A_up_bar}   xF (F) = {xF}")
    print()

    print("\n╔══ ANALISI TEORICA (valori attesi) ══╗")
    print("  n* calcolato con E[A_p] = A_p_bar*(1-p*gamma_p)*(1+n_p)")
    _print_mode_block("expected", kw)

    print("╔══ ANALISI SIMULATIVA (produttività reale) ══╗")
    print("  n* calcolato con la produttività effettiva.")
    print("  Ad ogni step la simulazione si trova in uno")
    print("  dei due casi seguenti con prob (1-p) e p.\n")
    for mode in SIMULATION_MODES:
        _print_mode_block(mode, kw)

    # riepilogo zone stocastiche
    ns = n_star_simulation(A_p_bar, mu=mu, r=r, alpha=alpha, gamma_p=gamma_p)
    ns_ns = ns["noshock"]
    ns_sh = ns["shock"]
    print("╔══ ZONE DI ALTERNANZA STOCASTICA ══╗")
    print(f"  (con p = {p}, gamma_p = {gamma_p}, gamma_up = {gamma_up})")
    if ns_ns < 0 < ns_sh < 1:
        print(f"  n^P ∈ (0,      {ns_sh:.3f}) → noshock esplosivo, "
              f"shock contrattivo  [alternanza]")
        print(f"  n^P ∈ ({ns_sh:.3f}, 1     ) → entrambi esplosivi "
              f"                   [sempre esplosivo]")
    elif 0 < ns_ns < ns_sh < 1:
        print(f"  n^P ∈ (0,      {ns_ns:.3f}) → entrambi contrattivi")
        print(f"  n^P ∈ ({ns_ns:.3f}, {ns_sh:.3f}) → noshock esplosivo, shock contrattivo")
        print(f"  n^P ∈ ({ns_sh:.3f}, 1     ) → entrambi esplosivi")
    else:
        print(f"  n*_noshock = {ns_ns:.4f},  n*_shock = {ns_sh:.4f}")
    print()