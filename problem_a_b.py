import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants given in the problem
SWEEPS = 20_000
H = 1
M = 1
TAU = 30
DELTATAU = 1
NTAU = int(TAU/DELTATAU)
HITSIZE = 0.1
ALPHA = 0.4  # Alpha for the double well potential

# Range for x values
XLOW = -4
XHIGH = 4
NXBINS = 100
DELTAX = (XHIGH - XLOW) / NXBINS
prob_histogram = np.zeros(NXBINS)
x_bins = np.linspace(XLOW, XHIGH, NXBINS + 1)

x_path = np.zeros(NTAU)

def V_double_well(x, alpha):
    """
    Double well potential function V(x) = alpha * x^4 - 2 * x^2 + 1/alpha
    """
    return alpha * x**4 - 2 * x**2 + 1/alpha

def vary_path(x_current, hit_size, xlow, xhigh):
    """
    Proposes a new path by varying the current path within the range of [xlow, xhigh].
    """
    x_prime = x_current + np.random.uniform(-hit_size, hit_size)
    while x_prime > xhigh or x_prime < xlow:
        x_prime = x_current + np.random.uniform(-hit_size, hit_size)
    return x_prime

def action(x_left, x_right, m, delta_tau, alpha):
    """
    Computes the action for a given slice of the path.
    """
    kinetic_term = 0.5 * m * ((x_right - x_left) / delta_tau)**2
    potential_term = V_double_well(0.5 * (x_left + x_right), alpha)
    return delta_tau * (kinetic_term + potential_term)

def delta_action(x_path, x_prime, i, m, delta_tau, alpha):
    """
    Computes the change in action delta S for a proposed path change at position i.
    """
    x_left = x_path[i-1]
    x_right = x_path[(i+1) % NTAU]  # Using modulo for periodic boundary condition
    S_old = action(x_left, x_right, m, delta_tau, alpha) + action(x_path[i], x_right, m, delta_tau, alpha)
    S_new = action(x_left, x_prime, m, delta_tau, alpha) + action(x_prime, x_right, m, delta_tau, alpha)
    return S_new - S_old

def MCMC(x_path, prob_histogram, m, delta_tau, alpha, hit_size, xlow, xhigh, nxbins):
    """
    Performs a single sweep of the Metropolis-Hastings algorithm over all time slices of the path.
    """
    for i in range(NTAU):
        x_prime = vary_path(x_path[i], hit_size, xlow, xhigh)
        dS = delta_action(x_path, x_prime, i, m, delta_tau, alpha)
        if dS <= 0 or np.random.rand() < np.exp(-dS):
            x_path[i] = x_prime
    
    # Update the histogram of the path positions
    hist, _ = np.histogram(x_path, bins=nxbins, range=(xlow, xhigh))
    prob_histogram += hist

# Running the MCMC simulation
for sweep in tqdm(range(SWEEPS), desc='MCMC Sweeps'):
    MCMC(x_path, prob_histogram, M, DELTATAU, ALPHA, HITSIZE, XLOW, XHIGH, NXBINS)

# Normalize the probability histogram
prob_histogram_normalized = prob_histogram / np.sum(prob_histogram) / DELTAX

# Check the normalization of the probability histogram
print(f'Probability normalization: {np.sum(prob_histogram_normalized) * DELTAX:.3f}')

# Plotting the probability distribution
plt.figure(figsize=(10, 6))
plt.stairs(prob_histogram_normalized, x_bins, label='MCMC for Double Well')
plt.title('Probability Distribution from MCMC Simulation')
plt.xlabel('x position')
plt.ylabel('Probability density')
plt.legend()
plt.show()

# Evaluate the corresponding ground state energy
E = 0
for i in range(NTAU):
    x_left = x_path[i-1]
    x_right = x_path[(i+1) % NTAU]
    E += 0.5 * M * ((x_path[i] - x_left) / DELTATAU)**2 + V_double_well(0.5 * (x_left + x_path[i]), ALPHA)

E /= NTAU
print(f'Ground state energy: {E:.3f}')

# Using assignment 3 to find the expected ground state
OMEGA = 1
BOXSIZE = 8
ND = 600
DELTAT = np.pi / 128
DELTAX = BOXSIZE / ND
HBAR = 1
ALPHA = 0.4

x = np.linspace(-BOXSIZE / 2, BOXSIZE / 2, ND + 1)

def V(x):
    return ALPHA*x**4 - 2*x**2 + 1/ALPHA

H = np.zeros((ND + 1, ND + 1))

for i in range(ND + 1):
    for j in range(ND + 1):
        # kinetic part
        H[i, j] = -(0.5 / DELTAX**2) * ((i + 1 == j) - 2 * (i == j) + (i - 1 == j)) 
        # potential part
        H[i, j] += V(x[i]) * (i == j)

# print the first 4x4 elements of H
print(H[:4,:4])

import scipy
def power_method(H, sigma, n_iter):
    Hms_inv = scipy.linalg.inv(H - sigma*np.eye(ND + 1))
    u = np.random.random(size=(ND + 1))
    lambda_u = np.dot(u.conjugate(), H @ u) / np.dot(u.conjugate(), u)
    for _ in range(n_iter):
        u = Hms_inv @ u
        u /= np.sqrt(np.dot(u.conjugate(),  u) * DELTAX)
        lambda_u = np.dot(u.conjugate(), H @ u) / np.dot(u.conjugate(), u)
    return lambda_u, u
    
E_0, psi_0 = power_method(H, 1.2, 100)
E_1, psi_1 = power_method(H, 1.4, 100)

# Find the ground state probability distribution
prob = np.abs(psi_0)**2
prob /= np.sum(prob) * DELTAX

# Plot the ground state probability distribution and compare with MCMC result
plt.figure(figsize=(10, 6))
plt.stairs(prob_histogram_normalized, x_bins, label='MCMC for Double Well')
plt.plot(x, prob, label='Expected')
plt.title('Ground State Probability Distribution')
plt.xlabel('x position')
plt.ylabel('Probability density')
plt.legend()
plt.show()