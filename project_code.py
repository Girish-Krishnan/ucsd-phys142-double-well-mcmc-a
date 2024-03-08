import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import numba # to accelerate the code

# Constants given in the problem
SWEEPS = 30_000
H = 1
M = 1
TAU = 50000
DELTATAU = 1
NTAU = int(TAU/DELTATAU)
HITSIZE = 0.1
ALPHA = 0.4  # Alpha for the double well potential
BURNIN = 5000

# Range for x values
XLOW = -4
XHIGH = 4
NXBINS = 100
DELTAX = (XHIGH - XLOW) / NXBINS
prob_histogram = np.zeros(NXBINS)
x_bins = np.linspace(XLOW, XHIGH, NXBINS + 1)

#x_path = np.zeros(NTAU)
# Instead of doing initialization with zeros, let's initialize randomly with either 1/sqrt(alpha) or -1/sqrt(alpha)
x_path = np.random.choice([-1/np.sqrt(ALPHA), 1/np.sqrt(ALPHA)], NTAU)

@numba.jit
def V_double_well(x, alpha):
    """
    Double well potential function V(x) = alpha * x^4 - 2 * x^2 + 1/alpha
    """
    return alpha * x**4 - 2 * x**2 + 1/alpha

@numba.jit
def vary_path(x_current, hit_size, xlow, xhigh):
    """
    Proposes a new path by varying the current path within the range of [xlow, xhigh].
    """
    x_prime = x_current + np.random.uniform(-hit_size, hit_size)
    while x_prime > xhigh or x_prime < xlow:
        x_prime = x_current + np.random.uniform(-hit_size, hit_size)
    return x_prime

@numba.jit
def action(x_left, x_right, m, delta_tau, alpha):
    """
    Computes the action for a given slice of the path.
    """
    kinetic_term = 0.5 * m * ((x_right - x_left) / delta_tau)**2
    potential_term = V_double_well(0.5 * (x_left + x_right), alpha)
    return delta_tau * (kinetic_term + potential_term)

@numba.jit
def delta_action(x_path, x_prime, i, m, delta_tau, alpha):
    """
    Computes the change in action delta S for a proposed path change at position i.
    """
    x_left = x_path[i-1]
    x_right = x_path[(i+1) % NTAU]  # Using modulo for periodic boundary condition
    S_old = action(x_left, x_path[i], m, delta_tau, alpha) + action(x_path[i], x_right, m, delta_tau, alpha)
    S_new = action(x_left, x_prime, m, delta_tau, alpha) + action(x_prime, x_right, m, delta_tau, alpha)
    return S_new - S_old

@numba.jit
def total_action(x_path, m, delta_tau, alpha, nx_bins):
    """
    Computes the total action for the entire path.
    """
    path_action = 0
    for i in range(-1, nx_bins - 1):
        path_action += action(x_path[i], x_path[i+1], m, delta_tau, alpha)

    return path_action

@numba.jit
def MCMC(x_path, m, delta_tau, alpha, hit_size, xlow, xhigh):
    """
    Performs a single sweep of the Metropolis-Hastings algorithm over all time slices of the path.
    """
    for i in range(NTAU):
        x_prime = vary_path(x_path[i], hit_size, xlow, xhigh)
        dS = delta_action(x_path, x_prime, i, m, delta_tau, alpha)
        if dS <= 0 or (np.random.random() < np.exp(-dS)):
            x_path[i] = x_prime
    
    # Update the histogram of the path positions
    hist, _ = np.histogram(x_path, bins=x_bins)
    return hist

# Running the MCMC simulation
for sweep in tqdm(range(SWEEPS), desc='MCMC Sweeps'):
    hist = MCMC(x_path, M, DELTATAU, ALPHA, HITSIZE, XLOW, XHIGH)
    if sweep > BURNIN:
        prob_histogram += hist

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

# Evaluate the corresponding ground state energy from the resulting probability distribution
H = np.zeros((NXBINS + 1, NXBINS + 1))
for i in range(NXBINS + 1):
    for j in range(NXBINS + 1):
        # kinetic part
        H[i, j] = -(0.5 / DELTAX**2) * ((i + 1 == j) - 2 * (i == j) + (i - 1 == j)) 
        # potential part
        H[i, j] += V_double_well(x_bins[i], ALPHA) * (i == j)

Es, psis = scipy.linalg.eig(H)
idx = np.argsort(Es)
Es = Es[idx]
psis = psis[:, idx]
E_ground = np.real(Es[0]) # Ground state energy

print(f'Ground state energy: {E_ground:.3f}')

# Using assignment 3 to find the expected ground state
BOXSIZE = 8
ND = 600
DELTAX = BOXSIZE / ND
x = np.linspace(-BOXSIZE / 2, BOXSIZE / 2, ND + 1)

H = np.zeros((ND + 1, ND + 1))

for i in range(ND + 1):
    for j in range(ND + 1):
        # kinetic part
        H[i, j] = -(0.5 / DELTAX**2) * ((i + 1 == j) - 2 * (i == j) + (i - 1 == j)) 
        # potential part
        H[i, j] += V_double_well(x[i], ALPHA) * (i == j)

# print the first 4x4 elements of H
print(H[:4,:4])

Es, psis = scipy.linalg.eig(H)
idx = np.argsort(Es)
Es = Es[idx]
psis = psis[:, idx]
psi_0 = psis[:, 0]

# Find the ground state probability distribution
prob = np.abs(psi_0)**2
prob /= np.sum(prob) * DELTAX

# Plot the ground state probability distribution and compare with MCMC result
plt.figure(figsize=(10, 6))
plt.stairs(prob_histogram_normalized, x_bins, label='MCMC for Double Well')
plt.plot(x, prob, label='Expected')
# Plot vertical lines at +/- 1/alpha to show the peaks
plt.axvline(-1/np.sqrt(ALPHA), color='r', linestyle='--', label='Expected Peaks')
plt.axvline(1/np.sqrt(ALPHA), color='r', linestyle='--')
plt.title('Ground State Probability Distribution')
plt.xlabel('x position')
plt.ylabel('Probability density')
plt.legend()
plt.show()