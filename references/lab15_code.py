import numpy as np
import matplotlib.pyplot as plt
import tqdm

SWEEPS = 20_000
H = 1
OMEGA = 1
M = 1
TAU = 30
DELTATAU = 1
NTAU = int(TAU/DELTATAU)
HITSIZE = 0.1

XLOW = -4
XHIGH = 4
NXBINS = 100
DELTAX = (XHIGH - XLOW) / NXBINS
HITSIZE = 0.1
prob_histogram = np.zeros(NXBINS)
x_bins = np.linspace(XLOW, XHIGH, NXBINS + 1)

x_path = np.zeros(NTAU)
## analytical solution
psi_analytical = (M*OMEGA/(np.pi*H))**0.25*np.exp(-M*OMEGA*x_bins**2/2/H)

def vary_path(x_current):
    x_prime = x_current + np.random.random() * 2 * HITSIZE - HITSIZE
    while x_prime > XHIGH or x_prime < XLOW:
        x_prime = x_current + np.random.random() * 2 * HITSIZE - HITSIZE
    return x_prime

def action(x_left, x_right):
    K = 0.5 * M * (((x_right - x_left))**2) / DELTATAU
    V = 0.5 * M * DELTATAU * (OMEGA**2) * (((x_left + x_right) / 2)**2)
    return K + V

def total_action(x_path):
    path_action = 0
    for i in range(-1, NXBINS-1):
        path_action += action(x_path[i], x_path[i+1])
    return path_action

def delta_action(x_path, x_prime, i):
    x_left = x_path[i-1]
    x_right = x_path[i+1] if i < NTAU-1 else x_path[0] #PBC.
    daction = action(x_left, x_prime) + action(x_prime, x_right) 
    daction -= action(x_left, x_path[i]) + action(x_path[i], x_right) #compute the resulting change from u in delta S.
    return daction

def MCMC(x_path, prob_histogram):
    for i in range(NTAU):
        x_prime = vary_path(x_path[i])
        daction = delta_action(x_path, x_prime, i)
        if daction <= 0:            
            x_path[i] = x_prime
        else:        
            prob = np.exp(-daction)
            if np.random.random() < prob:
                x_path[i] = x_prime
    hist, _ = np.histogram(x_path, bins=x_bins)
    prob_histogram += hist

for k in tqdm.tqdm(range(SWEEPS)):
    MCMC(x_path, prob_histogram)

plt.figure()
plt.stairs(prob_histogram/np.sum(prob_histogram*DELTAX), x_bins, label=f"MCMC, $\\tau = {TAU}$, $\\delta\\tau={DELTATAU}$")
plt.plot(x_bins, psi_analytical*psi_analytical.conjugate(), label="Analytical")
plt.legend()
plt.show()