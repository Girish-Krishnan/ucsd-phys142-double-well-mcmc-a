import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import numba # to accelerate the code

# Constants given in the problem
SWEEPS = 80000
H = 1
M = 1
TAU = 3000
DELTATAU = 0.1
NTAU = int(TAU/DELTATAU)
HITSIZE = 0.1
ALPHA = 0.4  # Alpha for the double well potential
BURNIN = 40000
THINNING = 50 # consider every 10th sample for the histogram to reduce correlation

# Range for x values
XLOW = -4
XHIGH = 4
NXBINS = 400
DELTAX = (XHIGH - XLOW) / NXBINS
prob_histogram = np.zeros(NXBINS)
x_bins = np.linspace(XLOW, XHIGH, NXBINS + 1)

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

@numba.jit
def Hamiltonian(nx_bins, x_bins, delta_x, alpha):
    """
    Returns the H matrix for the Hamiltonian.
    """
    H = np.zeros((nx_bins + 1, nx_bins + 1))
    for i in range(nx_bins + 1):
        for j in range(nx_bins + 1):
            # kinetic part
            H[i, j] = -(0.5 / delta_x**2) * ((i + 1 == j) - 2 * (i == j) + (i - 1 == j)) 
            # potential part
            H[i, j] += V_double_well(x_bins[i], alpha) * (i == j)

    return H

@numba.jit
def ground_state_energy(H, prob_histogram_normalized, delta_x):
    """
    Returns the ground state energy from the Hamiltonian and the probability distribution.
    """
    K = np.real((1 / 2)*np.sum(DELTAX * np.diff(np.sqrt(prob_histogram_normalized))*np.diff(np.sqrt(prob_histogram_normalized)) / DELTAX**2))
    V = np.sum(prob_histogram_normalized * V_double_well(x_bins[:-1], ALPHA)) * DELTAX
    return K + V

# Running the MCMC simulation
energy_values = []
action_values = []
hists = []
energy_calculation_interval = 100 # Calculate the ground state energy every 100 sweeps

for sweep in tqdm(range(SWEEPS), desc='MCMC Sweeps'):
    hist = MCMC(x_path, M, DELTATAU, ALPHA, HITSIZE, XLOW, XHIGH)
    if sweep % THINNING == 0:
        prob_histogram += hist
        hists.append(hist)

        if sweep % energy_calculation_interval == 0 and sweep != 0: # Calculate the ground state energy every energy_calculation_interval sweeps
            prob_histogram_normalized = prob_histogram / np.sum(prob_histogram) / DELTAX
            H = Hamiltonian(NXBINS, x_bins, DELTAX, ALPHA)
            energy_values.append(ground_state_energy(H, prob_histogram_normalized, DELTAX))
            action_values.append(total_action(x_path, M, DELTATAU, ALPHA, NXBINS))

# Take only the values after burn-in
prob_histogram = np.sum(hists[BURNIN//THINNING:], axis=0)

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

# Plot the energy as a function of the number of sweeps
plt.figure(figsize=(10, 6))
plt.plot(range(1, BURNIN, energy_calculation_interval), energy_values[:BURNIN//energy_calculation_interval], label='Burn-in', color='red')
plt.plot(range(BURNIN, SWEEPS, energy_calculation_interval), energy_values[BURNIN//energy_calculation_interval - 1:], label='After Burn-in', color='blue')
plt.plot([BURNIN - energy_calculation_interval, BURNIN], [energy_values[BURNIN//energy_calculation_interval-1], energy_values[BURNIN//energy_calculation_interval-1]], color='blue')
plt.title('Energy as a Function of Sweeps')
plt.xlabel('Sweeps')
plt.ylabel('Energy')
plt.legend()
plt.show()

# Plot the action as a function of the number of sweeps
plt.figure(figsize=(10, 6))
plt.plot(range(1, BURNIN, energy_calculation_interval), action_values[:BURNIN//energy_calculation_interval], label='Burn-in', color='red')
plt.plot(range(BURNIN, SWEEPS, energy_calculation_interval), action_values[BURNIN//energy_calculation_interval - 1:], label='After Burn-in', color='blue')
plt.plot([BURNIN - energy_calculation_interval, BURNIN], [action_values[BURNIN//energy_calculation_interval -1], action_values[BURNIN//energy_calculation_interval -1]], color='blue')
plt.title('Action as a Function of Sweeps')
plt.xlabel('Sweeps')
plt.ylabel('Action')
plt.legend()
plt.show()

# Evaluate the corresponding ground state energy from the resulting probability distribution
# Energy = <H>

# Use the Hamiltonian and the probability distribution to find the ground state energy
# based on the expectation value of the Hamiltonian matrix
H = Hamiltonian(NXBINS, x_bins, DELTAX, ALPHA)
E_ground = np.sum(np.sqrt(prob_histogram_normalized) @ H[:-1, :-1] @ np.sqrt(prob_histogram_normalized)) * DELTAX
# Since the wavefunction for ground state is real and positive, the square root of the PDF can be used

print("[Method 1] Ground state energy from MCMC simulation:", E_ground)

# Alternatively, calculate the ground state energy by doing <K> + <V>
K = np.real((1 / 2)*np.sum(DELTAX * np.diff(np.sqrt(prob_histogram_normalized))*np.diff(np.sqrt(prob_histogram_normalized)) / DELTAX**2))
V = np.sum(prob_histogram_normalized * V_double_well(x_bins[:-1], ALPHA)) * DELTAX
E_ground = K + V

print("[Method 2] Ground state energy from MCMC simulation:", E_ground)

# Using assignment 3 to find the expected ground state
BOXSIZE = 8
ND = 600
DELTAX = BOXSIZE / ND
x = np.linspace(-BOXSIZE / 2, BOXSIZE / 2, ND + 1)

def ground_state(H):
    """
    Returns the ground state energy and wave function.
    """
    Es, psis = scipy.linalg.eig(H)
    idx = np.argsort(Es)
    Es = Es[idx]
    psis = psis[:, idx]
    return np.real(Es[0]), psis[:, 0]

H = Hamiltonian(ND, x, DELTAX, ALPHA)
# print the first 4x4 elements of H
print('First 4x4 elements of H:')
print(H[:4,:4])

E_ground, psi_0 = ground_state(H)
# Find the ground state probability distribution
prob = np.abs(psi_0)**2
prob /= np.sum(prob) * DELTAX

print("Expected ground state energy:", E_ground)

# Plot the ground state probability distribution and compare with MCMC result
plt.figure(figsize=(10, 6))
plt.stairs(prob_histogram_normalized, x_bins, label='MCMC for Double Well')
plt.plot(x, prob, label='Expected')
plt.title('Ground State Probability Distribution')
plt.xlabel('x position')
plt.ylabel('Probability density')
plt.legend()
plt.show()

# Now, make a class for the MCMC simulation that allows for easy parameter changes and multiple runs
class DoubleWellMCMC:
    def __init__(self, sweeps, m, delta_tau, alpha, hit_size, xlow, xhigh, burnin, thinning, energy_calculation_interval, num_walkers, num_repeats):
        self.sweeps = sweeps
        self.m = m
        self.delta_tau = delta_tau
        self.alpha = alpha
        self.hit_size = hit_size
        self.xlow = xlow
        self.xhigh = xhigh
        self.burnin = burnin
        self.thinning = thinning
        self.energy_calculation_interval = energy_calculation_interval
        self.num_walkers = num_walkers
        self.num_repeats = num_repeats

    def run(self):
        prob_histograms = []
        energy_values = []
        action_values = []

        for repeat in range(self.num_repeats):
            prob_histogram = np.zeros(NXBINS)
            x_paths = np.random.choice([-1/np.sqrt(self.alpha), 1/np.sqrt(self.alpha)], (self.num_walkers, NTAU))

            for sweep in tqdm(range(self.sweeps), desc=f'MCMC Sweeps (Repeat {repeat+1}/{self.num_repeats})'):
                hists = []
                for walker in range(self.num_walkers):
                    hist = MCMC(x_paths[walker], self.m, self.delta_tau, self.alpha, self.hit_size, self.xlow, self.xhigh)
                    if sweep % self.thinning == 0:
                        prob_histogram += hist
                        hists.append(hist)

                if sweep % self.energy_calculation_interval == 0 and sweep != 0:
                    prob_histogram_normalized = prob_histogram / np.sum(prob_histogram) / DELTAX
                    H = Hamiltonian(NXBINS, x_bins, DELTAX, ALPHA)
                    energy_values.append(ground_state_energy(H, prob_histogram_normalized, DELTAX))
                    action_values.append(total_action(x_paths, M, DELTATAU, ALPHA, NXBINS))

            prob_histograms.append(np.sum(hists[self.burnin//self.thinning:], axis=0))

        prob_histograms = np.array(prob_histograms)
        prob_histogram_normalized = np.mean(prob_histograms, axis=0) / np.sum(np.mean(prob_histograms, axis=0)) / DELTAX

        E_ground = np.sum(np.sqrt(prob_histogram_normalized) @ H[:-1, :-1] @ np.sqrt(prob_histogram_normalized)) * DELTAX
        print("Ground state energy from MCMC simulation:", E_ground)

        return prob_histogram_normalized, energy_values, action_values, prob_histograms
    
    def plot_histogram(self, prob_histogram_normalized):
        plt.figure(figsize=(10, 6))
        plt.stairs(prob_histogram_normalized, x_bins, label='MCMC for Double Well')
        plt.title('Probability Distribution from MCMC Simulation')
        plt.xlabel('x position')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()

    def plot_energy(self, energy_values):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.burnin, self.energy_calculation_interval), energy_values[:self.burnin//self.energy_calculation_interval], label='Burn-in', color='red')
        plt.plot(range(self.burnin, self.sweeps, self.energy_calculation_interval), energy_values[self.burnin//self.energy_calculation_interval - 1:], label='After Burn-in', color='blue')
        plt.plot([self.burnin - self.energy_calculation_interval, self.burnin], [energy_values[self.burnin//self.energy_calculation_interval-1], energy_values[self.burnin//self.energy_calculation_interval-1]], color='blue')
        plt.title('Energy as a Function of Sweeps')
        plt.xlabel('Sweeps')
        plt.ylabel('Energy')
        plt.legend()
        plt.show()

    def plot_action(self, action_values):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.burnin, self.energy_calculation_interval), action_values[:self.burnin//self.energy_calculation_interval], label='Burn-in', color='red')
        plt.plot(range(self.burnin, self.sweeps, self.energy_calculation_interval), action_values[self.burnin//self.energy_calculation_interval - 1:], label='After Burn-in', color='blue')
        plt.plot([self.burnin - self.energy_calculation_interval, self.burnin], [action_values[self.burnin//self.energy_calculation_interval -1], action_values[self.burnin//self.energy_calculation_interval -1]], color='blue')
        plt.title('Action as a Function of Sweeps')
        plt.xlabel('Sweeps')
        plt.ylabel('Action')
        plt.legend()
        plt.show()

    def plot_histogram_variation(self, prob_histograms):
        plt.figure(figsize=(10, 6))
        for i in range(self.num_repeats):
            plt.stairs(prob_histograms[i] / np.sum(prob_histograms[i]) / DELTAX, x_bins, alpha=0.5, label=f'Walk {i+1}')
        plt.title('Variation in Probability Distribution for Multiple Walks')
        plt.xlabel('x position')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()

# Run the MCMC simulation with the class