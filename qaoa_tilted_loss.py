import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import networkx as nx
from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.primitives import StatevectorSampler

import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

class QAOAMaxCut:
    def __init__(self, graph, p=1, shots=2000):
        self.graph = graph
        self.number_of_qubits = graph.number_of_nodes()
        self.p = p
        self.shots = shots
        self.sampler = StatevectorSampler()
        
    def execute(self, params):
        qc = QuantumCircuit(self.number_of_qubits)
        
        for i in range(self.number_of_qubits):
            qc.h(i)
            
        for layer in range(self.p):
            gamma = params[layer]
            beta = params[self.p + layer]
            
            for i, j in self.graph.edges():
                qc.cx(i, j)
                qc.rz(gamma, j)
                qc.cx(i, j)
                
            for i in range(self.number_of_qubits):
                qc.rx(2 * beta, i)
                
        creg = ClassicalRegister(self.number_of_qubits, 'c')
        qc.add_register(creg)
        qc.measure(range(self.number_of_qubits), creg)
        
        pub = (qc,)
        job = self.sampler.run([pub], shots=self.shots)
        result = job.result()[0]
        self.counts = result.data.c.get_counts()
        return self.counts

    def cost_hamiltonian(self, x):
        cut_weight = 0
        for i, j in self.graph.edges():
            if x[i] != x[j]:
                cut_weight += 1
        return -cut_weight 

    def exact_counts(self):
        energies = []
        for sample, count in self.counts.items():
            y = [0] * self.number_of_qubits
            for i, char in enumerate(sample[::-1]):
                y[i] = int(char)
                
            tmp_eng = self.cost_hamiltonian(y)
            energies.extend([tmp_eng] * count)
        energies.sort(reverse=False)
        return energies

    def best_cost_brute(self):
        best_cost = np.inf
        for b in range(2**self.number_of_qubits):
            x = [int(t) for t in list(bin(b)[2:].zfill(self.number_of_qubits))[::-1]]
            cost = self.cost_hamiltonian(x)
            if cost < best_cost:
                best_cost = cost
        return best_cost

    def probability_of_optimal(self):
        optimal_solution = self.best_cost_brute()
        energies = self.exact_counts()
        return sum(1 for energy in energies if energy == optimal_solution) / self.shots


class QAOAOptimizerWrapper:
    def __init__(self, qaoa_inst, maxiter):
        self.qaoa = qaoa_inst
        self.maxiter = maxiter
        self.history = []
        self.eval_count = 0

    def reset(self):
        self.history = []
        self.eval_count = 0

    def get_standard_loss(self, angles):
        self.qaoa.execute(angles)
        energies = self.qaoa.exact_counts()
        energy_expectation = np.mean(energies)
        self.history.append(energy_expectation)
        return energy_expectation

    def get_CVaR_loss(self, angles, alpha):
        self.qaoa.execute(angles)
        energies = self.qaoa.exact_counts()
        num_samples = max(1, math.ceil(len(energies) * alpha))
        cvar = np.sum(energies[:num_samples]) / num_samples
        self.history.append(np.mean(energies)) # Store full expectation for fair plotting
        return cvar

    def get_Tilted_loss(self, angles, gamma):
        self.qaoa.execute(angles)
        energies = np.array(self.qaoa.exact_counts())
        full_exp = np.mean(energies)
        self.history.append(full_exp)
        
        if gamma == 0:
            return full_exp
            
        min_E = np.min(energies)
        exponent_terms = -gamma * (energies - min_E)
        log_term = np.log(np.mean(np.exp(exponent_terms)))
        return min_E - (1/gamma) * log_term

    def get_tuned_CVaR_loss(self, angles, options):
        T = options.get('max_evals', self.maxiter)
        alpha_start = options['alpha_start']
        alpha_end = options['alpha_end']
        schedule = options.get('schedule', 'linear')
        
        t = min(self.eval_count, T - 1)
        if T <= 1:
            alpha = alpha_start
        elif schedule == 'linear':
            alpha = alpha_start + (alpha_end - alpha_start) * (t / (T - 1))
        elif schedule == 'sigmoid':
            alpha = alpha_start + (alpha_end - alpha_start) / (1 + np.exp(-10 * (t / (T - 1) - 0.5)))
        elif schedule == 'exponential':
            alpha = alpha_start * (alpha_end / alpha_start)**(t / (T - 1))
        elif schedule == 'logarithmic':
            alpha = alpha_start + (alpha_end - alpha_start) * np.log(1 + t) / np.log(1 + T - 1)
        else:
            alpha = alpha_start
            
        loss = self.get_CVaR_loss(angles, alpha)
        self.eval_count += 1
        return loss

    def get_tuned_Tilted_loss(self, angles, options):
        T = options.get('max_evals', self.maxiter)
        gamma_start = options['gamma_start']
        gamma_end = options['gamma_end']
        schedule = options.get('schedule', 'linear')
        
        t = min(self.eval_count, T - 1)
        if T <= 1:
            gamma = gamma_start
        elif schedule == 'linear':
            gamma = gamma_start + (gamma_end - gamma_start) * (t / (T - 1))
        elif schedule == 'sigmoid':
            gamma = gamma_start + (gamma_end - gamma_start) / (1 + np.exp(-10 * (t / (T - 1) - 0.5)))
        elif schedule == 'exponential':
            gamma = gamma_start * (gamma_end / gamma_start)**(t / (T - 1))
        elif schedule == 'logarithmic':
            gamma = gamma_start + (gamma_end - gamma_start) * np.log(1 + t) / np.log(1 + T - 1)
        else:
            gamma = gamma_start

        loss = self.get_Tilted_loss(angles, gamma)
        self.eval_count += 1
        return loss

    def optimize(self, initial_angles, loss_type='standard', param=1.0, options=None):
        self.reset()
        if options is None:
            options = {}
            
        if loss_type == 'standard':
            opt = scipy.optimize.minimize(self.get_standard_loss, x0=tuple(initial_angles), method='COBYLA', options={'maxiter': self.maxiter})
            best_angles = opt.x
        elif loss_type == 'cvar':
            opt = scipy.optimize.minimize(self.get_CVaR_loss, x0=tuple(initial_angles), args=(param,), method='COBYLA', options={'maxiter': self.maxiter})
            best_angles = opt.x
        elif loss_type == 'tilted':
            opt = scipy.optimize.minimize(self.get_Tilted_loss, x0=tuple(initial_angles), args=(param,), method='COBYLA', options={'maxiter': self.maxiter})
            best_angles = opt.x
        elif loss_type == 'tuned_cvar':
            opt = scipy.optimize.minimize(self.get_tuned_CVaR_loss, x0=tuple(initial_angles), args=(options,), method='COBYLA', options={'maxiter': self.maxiter})
            best_angles = opt.x
        elif loss_type == 'tuned_tilted':
            opt = scipy.optimize.minimize(self.get_tuned_Tilted_loss, x0=tuple(initial_angles), args=(options,), method='COBYLA', options={'maxiter': self.maxiter})
            best_angles = opt.x
            
        self.qaoa.execute(best_angles)
        prob_optimal = self.qaoa.probability_of_optimal()
        return self.history.copy(), prob_optimal


def main():
    print("Generating Graph for MaxCut...")
    # Generate a random 6-node graph
    G = nx.erdos_renyi_graph(6, 0.6, seed=42)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    p = 2 # QAOA layers
    maxiter = 50
    init_thetas = [np.random.uniform(0, 2*np.pi) for _ in range(2 * p)] # [gamma_1, gamma_2, beta_1, beta_2]
    
    qaoa_inst = QAOAMaxCut(G, p=p, shots=2000)
    opt_wrapper = QAOAOptimizerWrapper(qaoa_inst, maxiter=maxiter)

    # 3. Compare Standard Loss, CVaR loss, and Tilted Loss
    print("\n--- Task 3: Compare Core Loss Functions ---")
    print("Optimizing Standard Loss...")
    hist_std, prob_std = opt_wrapper.optimize(init_thetas, loss_type='standard')
    
    print("Optimizing CVaR Loss (alpha=0.25)...")
    hist_cvar, prob_cvar = opt_wrapper.optimize(init_thetas, loss_type='cvar', param=0.25)
    
    print("Optimizing Tilted Loss (gamma=5.0)...")
    hist_tilted, prob_tilted = opt_wrapper.optimize(init_thetas, loss_type='tilted', param=5.0)

    print("\nProbabilities of Sampling Optimal Solution:")
    print(f"Standard Loss    : {prob_std:.4f}")
    print(f"CVaR (alpha=0.25): {prob_cvar:.4f}")
    print(f"Tilted (gamma=5) : {prob_tilted:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(hist_std, label='Standard Loss', alpha=0.8)
    plt.plot(hist_cvar, label='CVaR (alpha=0.25)', alpha=0.8)
    plt.plot(hist_tilted, label='Tilted Loss (gamma=5)', alpha=0.8)
    plt.xlabel('Function Evaluations')
    plt.ylabel('Expectation Value <H>')
    plt.title('Comparison: expected Energy during Optimization')
    plt.legend()
    plt.grid(True)
    plt.savefig('task3_core_comparison.png')
    plt.close()

    # 4. Compare fixed gamma values
    print("\n--- Task 4: Compare Fixed Gamma Values ---")
    gamma_values = [0.1, 5, 10, 20, 100]
    gamma_histories = {}
    gamma_probs = {}
    
    for gamma in gamma_values:
        print(f"Optimizing Tilted Loss with gamma={gamma}...")
        hist, prob = opt_wrapper.optimize(init_thetas, loss_type='tilted', param=gamma)
        gamma_histories[gamma] = hist
        gamma_probs[gamma] = prob
        
    print("\nProbabilities of Sampling Optimal Solution (Fixed Gammas):")
    for gamma in gamma_values:
        print(f"gamma={gamma:<5} : {gamma_probs[gamma]:.4f}")

    plt.figure(figsize=(10, 6))
    for gamma in gamma_values:
        plt.plot(gamma_histories[gamma], label=f'gamma={gamma}', alpha=0.8)
    plt.xlabel('Function Evaluations')
    plt.ylabel('Expectation Value <H>')
    plt.title('Tilted Loss Performance across Fixed Gamma values')
    plt.legend()
    plt.grid(True)
    plt.savefig('task4_fixed_gammas.png')
    plt.close()

    # 5. Tune gamma schedules and compare with tuned CVaR
    print("\n--- Task 5: Tuning Gamma Schedules vs Ascending CVaR ---")
    
    # We will tune at EVERY ITERATION STEP. 
    # For CVaR, let's start alpha from 0.05 and ascend to 1.0 (Ascending CVaR)
    cvar_options = {'max_evals': maxiter, 'alpha_start': 0.05, 'alpha_end': 1.0, 'schedule': 'linear'}
    print("Optimizing Tuned CVaR (Ascending)...")
    hist_asc_cvar, prob_asc_cvar = opt_wrapper.optimize(init_thetas, loss_type='tuned_cvar', options=cvar_options)
    
    schedules = ['linear', 'sigmoid', 'exponential', 'logarithmic']
    gamma_start = 0.1
    gamma_end = 20.0
    sched_histories = {}
    sched_probs = {}

    for sched in schedules:
        print(f"Optimizing Tuned Tilted Loss with schedule: {sched}...")
        opts = {
            'max_evals': maxiter,
            'schedule': sched,
            'gamma_start': gamma_start,
            'gamma_end': gamma_end
        }
        hist, prob = opt_wrapper.optimize(init_thetas, loss_type='tuned_tilted', options=opts)
        sched_histories[sched] = hist
        sched_probs[sched] = prob

    print("\nProbabilities of Sampling Optimal Solution (Tuning Schedules):")
    print(f"Tuned (Ascending) CVaR    : {prob_asc_cvar:.4f}")
    for sched in schedules:
        print(f"Tilted Tuned ({sched:<11}) : {sched_probs[sched]:.4f}")

    plt.figure(figsize=(12, 7))
    plt.plot(hist_asc_cvar, label='Ascending CVaR', color='black', linewidth=2, linestyle='--')
    for sched in schedules:
        plt.plot(sched_histories[sched], label=f'Tilted (sched={sched})', alpha=0.8)
    plt.xlabel('Function Evaluations')
    plt.ylabel('Expectation Value <H>')
    plt.title('Gamma Schedules vs Ascending CVaR (Tuning at Every Iteration Step)')
    plt.legend()
    plt.grid(True)
    plt.savefig('task5_tuning_schedules.png')
    plt.close()

if __name__ == '__main__':
    main()
