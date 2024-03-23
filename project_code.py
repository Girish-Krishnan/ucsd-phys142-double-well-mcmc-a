import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import numba # to accelerate the code
from matplotlib.animation import FuncAnimation

# Constants given in the problem
SWEEPS = 80000
H = 1
M = 1
TAU = 30
DELTATAU = 0.01
NTAU = int(TAU/DELTATAU)
HITSIZE = 0.5
ALPHA = 0.4  # Alpha for the double well potential
BURNIN = 40000
THINNING = 10 # consider every 10th sample for the histogram to reduce correlation

# Range for x values
XLOW = -4
XHIGH = 4
NXBINS = 400
DELTAX = (XHIGH - XLOW) / NXBINS
prob_histogram = np.zeros(NXBINS)
x_bins = np.linspace(XLOW, XHIGH, NXBINS + 1)

# Instead of doing initialization with zeros, let's initialize randomly with either 1/sqrt(alpha) or -1/sqrt(alpha)
x_path = np.random.choice([-1/np.sqrt(ALPHA), 1/np.sqrt(ALPHA)], NTAU)

# Alternatively, initialize the path in the same way as above, but make sure there's an equal number of 1/sqrt(alpha) and -1/sqrt(alpha)
# x_path = np.append([1/np.sqrt(ALPHA)] * (NTAU//2), [-1/np.sqrt(ALPHA)] * (NTAU//2))
# x_path = np.random.permutation(x_path)

# Alternatively, initialize the path with alternating 1/sqrt(alpha) and -1/sqrt(alpha)
x_path = np.array([1/np.sqrt(ALPHA) if i % 2 == 0 else -1/np.sqrt(ALPHA) for i in range(NTAU)])

# Alternatively, let around 1/4 of the path be zeros, and the remaining 3/4ths be split between 1/sqrt(alpha) and -1/sqrt(alpha)
# x_path = np.append([0] * (NTAU//4), [1/np.sqrt(ALPHA)] * (NTAU//4) + [-1/np.sqrt(ALPHA)] * (NTAU//4))


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
def delta_action(x_path, x_prime, i, m, delta_tau, alpha, n_tau):
    """
    Computes the change in action delta S for a proposed path change at position i.
    """
    x_left = x_path[i-1]
    x_right = x_path[(i+1) % n_tau]  # Using modulo for periodic boundary condition
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
def MCMC(x_path, m, delta_tau, alpha, hit_size, xlow, xhigh, n_tau):
    """
    Performs a single sweep of the Metropolis-Hastings algorithm over all time slices of the path.
    """
    for i in range(n_tau):
        x_prime = vary_path(x_path[i], hit_size, xlow, xhigh)
        dS = delta_action(x_path, x_prime, i, m, delta_tau, alpha, n_tau)
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
def ground_state_energy(prob_histogram_normalized, delta_x, alpha):
    """
    Returns the ground state energy from the Hamiltonian and the probability distribution.
    """
    K = np.real((1 / 2)*np.sum(delta_x * np.diff(np.sqrt(prob_histogram_normalized))*np.diff(np.sqrt(prob_histogram_normalized)) / delta_x**2))
    V = np.sum(prob_histogram_normalized * V_double_well(x_bins[:-1], alpha)) * delta_x
    return K + V

# Running the MCMC simulation
# energy_values = []
# action_values = []
# hists = []
# energy_calculation_interval = 100 # Calculate the ground state energy every 100 sweeps

# for sweep in tqdm(range(SWEEPS), desc='MCMC Sweeps'):
#     hist = MCMC(x_path, M, DELTATAU, ALPHA, HITSIZE, XLOW, XHIGH, NTAU)
#     if sweep % THINNING == 0:
#         prob_histogram += hist
#         hists.append(hist)

#         if sweep % energy_calculation_interval == 0 and sweep != 0: # Calculate the ground state energy every energy_calculation_interval sweeps
#             prob_histogram_normalized = prob_histogram / np.sum(prob_histogram) / DELTAX
#             energy_values.append(ground_state_energy(prob_histogram_normalized, DELTAX, ALPHA))
#             action_values.append(total_action(x_path, M, DELTATAU, ALPHA, NXBINS))

# # Make an animation of the normalized probability histogram over time
# # def update(i):
# #     plt.cla()
# #     plt.stairs(np.sum(hists[:i], axis=0) / np.sum(np.sum(hists[:i], axis=0)) / DELTAX, x_bins, label='MCMC for Double Well')
# #     plt.title('Probability Distribution from MCMC Simulation at Sweep ' + str(i))
# #     plt.xlabel('x position')
# #     plt.ylabel('Probability density')
# #     plt.legend()

# # fig = plt.figure(figsize=(10, 6))
# # ani = FuncAnimation(fig, update, frames=range(1, SWEEPS, 100), blit=False)
# # ani.save('mcmc_double_well.gif', writer='imagemagick', fps=10)

# # Take only the values after burn-in
# prob_histogram = np.sum(hists[BURNIN//THINNING:], axis=0)

# # Normalize the probability histogram
# prob_histogram_normalized = prob_histogram / np.sum(prob_histogram) / DELTAX

# # Check the normalization of the probability histogram
# print(f'Probability normalization: {np.sum(prob_histogram_normalized) * DELTAX:.3f}')

# # Plotting the probability distribution
# plt.figure(figsize=(10, 6))
# plt.stairs(prob_histogram_normalized, x_bins, label='MCMC for Double Well')
# plt.title('Probability Distribution from MCMC Simulation')
# plt.xlabel('x position')
# plt.ylabel('Probability density')
# plt.legend()
# plt.show()

# # Plot the energy as a function of the number of sweeps
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, BURNIN, energy_calculation_interval), energy_values[:BURNIN//energy_calculation_interval], label='Burn-in', color='red')
# plt.plot(range(BURNIN, SWEEPS, energy_calculation_interval), energy_values[BURNIN//energy_calculation_interval - 1:], label='After Burn-in', color='blue')
# plt.plot([BURNIN - energy_calculation_interval, BURNIN], [energy_values[BURNIN//energy_calculation_interval-1], energy_values[BURNIN//energy_calculation_interval-1]], color='blue')
# plt.title('Energy as a Function of Sweeps')
# plt.xlabel('Sweeps')
# plt.ylabel('Energy')
# plt.legend()
# plt.show()

# # Plot the action as a function of the number of sweeps
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, BURNIN, energy_calculation_interval), action_values[:BURNIN//energy_calculation_interval], label='Burn-in', color='red')
# plt.plot(range(BURNIN, SWEEPS, energy_calculation_interval), action_values[BURNIN//energy_calculation_interval - 1:], label='After Burn-in', color='blue')
# plt.plot([BURNIN - energy_calculation_interval, BURNIN], [action_values[BURNIN//energy_calculation_interval -1], action_values[BURNIN//energy_calculation_interval -1]], color='blue')
# plt.title('Action as a Function of Sweeps')
# plt.xlabel('Sweeps')
# plt.ylabel('Action')
# plt.legend()
# plt.show()

# # Evaluate the corresponding ground state energy from the resulting probability distribution
# # Energy = <H>

# # Use the Hamiltonian and the probability distribution to find the ground state energy
# # based on the expectation value of the Hamiltonian matrix
# H = Hamiltonian(NXBINS, x_bins, DELTAX, ALPHA)
# E_ground = np.sum(np.sqrt(prob_histogram_normalized) @ H[:-1, :-1] @ np.sqrt(prob_histogram_normalized)) * DELTAX
# # Since the wavefunction for ground state is real and positive, the square root of the PDF can be used

# print("[Method 1] Ground state energy from MCMC simulation:", E_ground)

# # Alternatively, calculate the ground state energy by doing <K> + <V>
# K = np.real((1 / 2)*np.sum(DELTAX * np.diff(np.sqrt(prob_histogram_normalized))*np.diff(np.sqrt(prob_histogram_normalized)) / DELTAX**2))
# V = np.sum(prob_histogram_normalized * V_double_well(x_bins[:-1], ALPHA)) * DELTAX
# E_ground = K + V

# print("[Method 2] Ground state energy from MCMC simulation:", E_ground)

# # Using assignment 3 to find the expected ground state
# BOXSIZE = 8
# ND = 600
# DELTAX = BOXSIZE / ND
# x = np.linspace(-BOXSIZE / 2, BOXSIZE / 2, ND + 1)

def ground_state(H):
    """
    Returns the ground state energy and wave function.
    """
    Es, psis = scipy.linalg.eig(H)
    idx = np.argsort(Es)
    Es = Es[idx]
    psis = psis[:, idx]
    return np.real(Es[0]), psis[:, 0]

# H = Hamiltonian(ND, x, DELTAX, ALPHA)
# # print the first 4x4 elements of H
# print('First 4x4 elements of H:')
# print(H[:4,:4])

# E_ground, psi_0 = ground_state(H)
# # Find the ground state probability distribution
# prob = np.abs(psi_0)**2
# prob /= np.sum(prob) * DELTAX

# print("Expected ground state energy:", E_ground)

# # Plot the ground state probability distribution and compare with MCMC result
# plt.figure(figsize=(10, 6))
# plt.stairs(prob_histogram_normalized, x_bins, label='MCMC for Double Well')
# plt.plot(x, prob, label='Expected')
# plt.title('Ground State Probability Distribution')
# plt.xlabel('x position')
# plt.ylabel('Probability density')
# plt.legend()
# plt.show()

# Now, make a class for the MCMC simulation that allows for easy parameter changes and multiple runs
class DoubleWellMCMC:
    def __init__(self, sweeps, m, delta_tau, alpha, hit_size, xlow, xhigh, burnin, thinning, energy_calculation_interval, ntau, nxbins, x_bins, deltax):
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
        self.ntau = ntau
        self.nxbins = nxbins
        self.x_bins = x_bins
        self.deltax = deltax

    def run(self):
        # Initialize the path
        x_path = np.random.choice([-1/np.sqrt(self.alpha), 1/np.sqrt(self.alpha)], self.ntau)

        # Running the MCMC simulation
        energy_values = []
        action_values = []
        hists = []
        prob_histogram = np.zeros(self.nxbins)
        for sweep in tqdm(range(self.sweeps), desc='MCMC Sweeps'):
            hist = MCMC(x_path, self.m, self.delta_tau, self.alpha, self.hit_size, self.xlow, self.xhigh, self.ntau)
            if sweep % self.thinning == 0:
                prob_histogram += hist
                hists.append(hist)

                if sweep % self.energy_calculation_interval == 0 and sweep != 0:
                    prob_histogram_normalized = prob_histogram / np.sum(prob_histogram) / self.deltax
                    energy_values.append(ground_state_energy(prob_histogram_normalized, self.deltax, self.alpha))
                    action_values.append(total_action(x_path, self.m, self.delta_tau, self.alpha, self.nxbins))

        # Take only the values after burn-in
        prob_histogram = np.sum(hists[self.burnin//self.thinning:], axis=0)

        # Normalize the probability histogram
        prob_histogram_normalized = prob_histogram / np.sum(prob_histogram) / self.deltax

        return prob_histogram_normalized, energy_values, action_values
    
    def find_energy(self, prob_histogram_normalized):
        return ground_state_energy(prob_histogram_normalized, self.deltax, self.alpha)
    
    def find_expected_ground_state(self):
        # USe scipy to find the expected ground state
        H = Hamiltonian(self.nxbins, self.x_bins, self.deltax, self.alpha)
        E_ground, psi_0 = ground_state(H)
        prob = np.abs(psi_0)**2
        prob /= np.sum(prob) * self.deltax
        return E_ground, prob

    def plot_prob_distribution(self, prob_histogram_normalized):
        plt.figure(figsize=(10, 6))
        plt.stairs(prob_histogram_normalized, self.x_bins)
        plt.title('Probability Distribution from MCMC Simulation, Energy = ' + str(round(self.find_energy(prob_histogram_normalized), 3)))
        plt.xlabel('x position')
        plt.ylabel('Probability density')
        plt.show()

    def plot_prob_and_expected(self, prob_histogram_normalized, prob_expected):
        plt.figure(figsize=(10, 6))
        plt.stairs(prob_histogram_normalized, self.x_bins, label='MCMC for Double Well')
        plt.plot(self.x_bins, prob_expected, label='Expected')
        plt.title('Ground State Probability Distribution')
        plt.xlabel('x position')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()

    def plot_energy(self, energy_values):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.burnin, self.energy_calculation_interval), energy_values[:self.burnin//self.energy_calculation_interval], label='Burn-in', color='red')
        plt.plot(range(self.burnin, self.sweeps, self.energy_calculation_interval), energy_values[self.burnin//self.energy_calculation_interval - 1:], label='After Burn-in', color='blue')
        plt.plot([self.burnin - self.energy_calculation_interval, self.burnin], [energy_values[self.burnin//self.energy_calculation_interval-1], energy_values[self.burnin//self.energy_calculation_interval-1]], color='blue')
        # Find the expected ground state
        E_ground_expected, prob_expected = self.find_expected_ground_state()
        # Plot a horizontal line at the expected ground state energy
        plt.axhline(y=E_ground_expected, color='green', linestyle='--', label='Expected Ground State Energy')
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


# Run the MCMC simulation with the class
energy_calculation_interval = 100 # Calculate the ground state energy every 100 sweeps
# mcmc = DoubleWellMCMC(SWEEPS, M, DELTATAU, ALPHA, HITSIZE, XLOW, XHIGH, BURNIN, THINNING, energy_calculation_interval, NTAU, NXBINS, x_bins, DELTAX)
# prob_histogram_normalized, energy_values, action_values = mcmc.run()
# mcmc.plot_prob_distribution(prob_histogram_normalized)
# mcmc.plot_energy(energy_values)
# mcmc.plot_action(action_values)

# # Find the expected ground state
# E_ground_expected, prob_expected = mcmc.find_expected_ground_state()
# print("Expected ground state energy:", E_ground_expected)

# # Find the ground state energy from the MCMC simulation
# E_ground_mcmc = mcmc.find_energy(prob_histogram_normalized)
# print("Ground state energy from MCMC simulation:", E_ground_mcmc)

# # Plot the ground state probability distribution and compare with MCMC result
# mcmc.plot_prob_and_expected(prob_histogram_normalized, prob_expected)

# Now, for the smaller tau value, find the expected energy
# In this case, it hasn't reached ground state yet, so we consider a superposition of the first two states

# Make an animation of the normalized probability histogram over time
# def update(i):
#     plt.cla()
#     plt.stairs(np.sum(hists[:i], axis=0) / np.sum(np.sum(hists[:i], axis=0)) / DELTAX, x_bins, label='MCMC for Double Well')
#     plt.title('Probability Distribution from MCMC Simulation at Sweep ' + str(i))
#     plt.xlabel('x position')
#     plt.ylabel('Probability density')
#     plt.legend()

# fig = plt.figure(figsize=(10, 6))
# ani = FuncAnimation(fig, update, frames=range(1, SWEEPS, 100), blit=False)
# ani.save('mcmc_double_well.gif', writer='imagemagick', fps=10)

# Now, run the MCMC experiment 5 times and calculate the mean histogram and energy
tau_values = [0.3, 3, 30, 300, 3000]
tau_hists = []
tau_energy = []

for tau in tau_values:
    print("Running MCMC simulation for tau =", tau)
    
    prob_histograms = []
    energy_values = []
    num_times = 5
    delta_tau = 0.01

    if tau == 300:
        num_times = 2

    if tau == 3000:
        num_times = 1
        delta_tau = 0.1

    NTAU = int(tau/delta_tau)

    for i in range(num_times):
        mcmc = DoubleWellMCMC(SWEEPS, M, DELTATAU, ALPHA, HITSIZE, XLOW, XHIGH, BURNIN, THINNING, energy_calculation_interval, NTAU, NXBINS, x_bins, DELTAX)
        prob_histogram_normalized, energy_values_single, _ = mcmc.run()
        prob_histograms.append(prob_histogram_normalized)
        energy_values.append(energy_values_single)

    # Calculate the mean histogram and energy
    mean_prob_histogram = np.mean(prob_histograms, axis=0)
    mean_energy = np.mean(energy_values, axis=0)
    tau_hists.append(mean_prob_histogram)
    tau_energy.append(mean_energy)
    # mcmc.plot_prob_distribution(mean_prob_histogram)
    # mcmc.plot_energy(mean_energy)
    # prob_expected = np.mean([mcmc.find_expected_ground_state()[1] for _ in range(1)], axis=0)
    # mcmc.plot_prob_and_expected(mean_prob_histogram, prob_expected)
    # E_ground_mcmc = mcmc.find_energy(mean_prob_histogram)
    # print("Ground state energy from MCMC simulation:", E_ground_mcmc)

# Plot the energy as a function of tau
plt.figure(figsize=(10, 6))
for i, tau in enumerate(tau_values):
    plt.plot(range(1, BURNIN, energy_calculation_interval), tau_energy[i][:BURNIN//energy_calculation_interval], label='Tau = ' + str(tau))
# Plot a horizontal line at the expected ground state energy
E_ground_expected, _ = mcmc.find_expected_ground_state()
plt.axhline(y=E_ground_expected, color='green', linestyle='--', label='Expected Ground State Energy = ' + str(round(E_ground_expected, 3)))
plt.title('Energy as a Function of Sweeps')
plt.xlabel('Sweeps')
plt.ylabel('Energy')
plt.legend(loc='upper right')
plt.show()

# Plot the probability distribution for different tau values
plt.figure(figsize=(10, 6))
for i, tau in enumerate(tau_values):
    plt.stairs(tau_hists[i], x_bins, label='Tau = ' + str(tau) + " Energy = " + str(round(tau_energy[i][-1], 3)))
# Plot the expected probability distribution
_, prob_expected = mcmc.find_expected_ground_state()
plt.plot(x_bins, prob_expected, label='Expected Ground State Probability Distribution')
plt.title('Probability Distribution from MCMC Simulation')
plt.xlabel('x position')
plt.ylabel('Probability density')
plt.legend(loc='upper right')
plt.show()

# Plot energy as a function of tau
plt.figure(figsize=(10, 6))
plt.plot(tau_values, [energy[-1] for energy in tau_energy], marker='o')
plt.title('Final Energy as a Function of Tau')
plt.xlabel('Tau')
plt.ylabel('Energy')
plt.show()