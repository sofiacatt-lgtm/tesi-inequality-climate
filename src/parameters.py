# parameters.py

import numpy as np

# Orizzonte temporale e numero agenti
T = 20
N = 50000

# Parametri economici
alpha = 0.65
mu = 2.0
r = 0.01
Ybar = 0.5
A_p_bar = 1.7
A_up_bar = 1.2

# Shock
p = 0.5
q = 0.5
gamma_p = 0.4
gamma_up = 0.6

eta_p = p * gamma_p
eta_up = q * gamma_up

# Soglie
F = 1.0
xF = F / (1 + mu)

# Logit
beta = 1
delta = 0.1
