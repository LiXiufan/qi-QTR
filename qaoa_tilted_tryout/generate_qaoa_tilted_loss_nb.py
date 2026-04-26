import nbformat as nbf
import os

def generate_notebook():
    nb = nbf.v4.new_notebook()

    # Introduction
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# Quantum Tilted Loss in QAOA and HEA for MaxCut\n\n"
        "This notebook explores the performance of the **Quantum Tilted Loss Function** in solving the MaxCut problem "
        "using the **Quantum Approximate Optimization Algorithm (QAOA)** and a **Hardware Efficient Ansatz (HEA)**.\n\n"
        "### 1. Mathematical Background\n"
        "The standard loss function in VQA is the expectation value of the Hamiltonian:\n"
        "$$\\mathcal{L}_{std}(\\theta) = \\bra{\\psi(\\theta)} H \\ket{\\psi(\\theta)}$$\n\n"
        "The **Quantum Tilted Loss Function** is defined as:\n"
        "$$\\mathcal{L}_\\gamma(H,\\rho(\\theta)) := - \\frac{1}{\gamma} \\log \\text{tr} \\left( e^{- \\gamma H} \\rho(\\theta) \\right)$$\n"
        "where $\gamma$ is a real parameter. As $\gamma \\to 0$, it recovers the standard expectation value. "
        "For $\gamma > 0$, it penalizes high-energy (sub-optimal) states more heavily, potentially accelerating convergence to the ground state.\n\n"
        "### 2. Goal\n"
        "1. Compare Standard Loss vs Tilted Loss for MaxCut with QAOA.\n"
        "2. Test Tilted Loss with Hardware Efficient Ansatz.\n"
        "3. Compare the algorithm performance with different fixed values of $\gamma$ (0.1, 5, 10, 20, 100).\n"
        "4. Investigate the effect of tuning $\gamma$ using different schedules (Linear, Sigmoid, Exponential, Logarithmic)."
    ))

    # Imports
    nb.cells.append(nbf.v4.new_code_cell(
        "import pennylane as qml\n"
        "from pennylane import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from scipy.optimize import minimize\n"
        "import time\n\n"
        "# Set random seed for reproducibility\n"
        "np.random.seed(42)"
    ))

    # Problem Definition (MaxCut)
    nb.cells.append(nbf.v4.new_code_cell(
        "# Define a small graph for MaxCut (4-node ring)\n"
        "nodes = 4\n"
        "edges = [(0, 1), (1, 2), (2, 3), (3, 0)]\n\n"
        "# Hamiltonian: H = sum 0.5 * (Zi * Zj - 1) for (i,j) in edges\n"
        "# Minimizing this minimizes edges with same spin, thus maximizing the cut.\n"
        "coeffs = [0.5] * len(edges) + [-0.5] * len(edges)\n"
        "obs = [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in edges] + [qml.Identity(i) for i in range(len(edges))]\n"
        "H = qml.Hamiltonian(coeffs, obs)\n\n"
        "print(f'Hamiltonian for {nodes}-node MaxCut:\\n', H)"
    ))

    # Circuit Definitions (QAOA and HEA)
    nb.cells.append(nbf.v4.new_code_cell(
        "dev = qml.device('default.qubit', wires=nodes)\n\n"
        "# QAOA Ansatz\n"
        "def qaoa_layer(gamma, beta, edges):\n"
        "    for i, j in edges:\n"
        "        qml.IsingZZ(2 * gamma, wires=[i, j])\n"
        "    for i in range(nodes):\n"
        "        qml.RX(2 * beta, wires=i)\n\n"
        "@qml.qnode(dev)\n"
        "def qaoa_circuit(params, p=1):\n"
        "    # Start in superposition\n"
        "    for i in range(nodes):\n"
        "        qml.Hadamard(wires=i)\n"
        "    # p layers of QAOA\n"
        "    alphas = params[:p]\n"
        "    betas = params[p:]\n"
        "    for i in range(p):\n"
        "        qaoa_layer(alphas[i], betas[i], edges)\n"
        "    return qml.probs(wires=range(nodes))\n\n"
        "# Hardware Efficient Ansatz (HEA)\n"
        "@qml.qnode(dev)\n"
        "def hea_circuit(params, layers=2):\n"
        "    # params shape: (layers, nodes, 2) for RY and RZ\n"
        "    idx = 0\n"
        "    for L in range(layers):\n"
        "        for i in range(nodes):\n"
        "            qml.RY(params[idx], wires=i); idx += 1\n"
        "            qml.RZ(params[idx], wires=i); idx += 1\n"
        "        # Entanglement (ring of CNOTs)\n"
        "        for i in range(nodes):\n"
        "            qml.CNOT(wires=[i, (i + 1) % nodes])\n"
        "    # Final rotation layer\n"
        "    for i in range(nodes):\n"
        "        qml.RY(params[idx], wires=i); idx += 1\n"
        "    return qml.probs(wires=range(nodes))"
    ))

    # Loss Functions
    nb.cells.append(nbf.v4.new_code_cell(
        "def get_energies(nodes, edges):\n"
        "    # Precompute eigenvalues of the MaxCut Hamiltonian for all basis states\n"
        "    energies = []\n"
        "    for i in range(2**nodes):\n"
        "        bitstring = format(i, f'0{nodes}b')\n"
        "        spins = [1 if b == '0' else -1 for b in bitstring]\n"
        "        energy = 0\n"
        "        for u, v in edges:\n"
        "            energy += 0.5 * (spins[u] * spins[v] - 1)\n"
        "        energies.append(energy)\n"
        "    return np.array(energies)\n\n"
        "energies = get_energies(nodes, edges)\n\n"
        "def standard_loss(probs):\n"
        "    return np.dot(probs, energies)\n\n"
        "def tilted_loss(probs, gamma_val):\n"
        "    # Avoid gamma = 0 to prevent division by zero; use standard limit\n"
        "    if abs(gamma_val) < 1e-9:\n"
        "        return standard_loss(probs)\n"
        "    # tr(exp(-gamma*H) * rho) = sum(p_i * exp(-gamma * E_i))\n"
        "    expectation_exp = np.dot(probs, np.exp(-gamma_val * energies))\n"
        "    return - (1.0 / gamma_val) * np.log(expectation_exp)"
    ))

    # Task 1: QAOA Comparison
    nb.cells.append(nbf.v4.new_markdown_cell("## Task 1: QAOA Comparison (Standard vs Tilted)"))
    nb.cells.append(nbf.v4.new_code_cell(
        "p = 2\n"
        "init_params = np.random.uniform(0, np.pi, 2 * p, requires_grad=True)\n"
        "gamma_fixed = 0.5\n\n"
        "history_std = []\n"
        "history_tilted = []\n\n"
        "def cost_std(params):\n"
        "    probs = qaoa_circuit(params, p=p)\n"
        "    loss = standard_loss(probs)\n"
        "    history_std.append(loss)\n"
        "    return loss\n\n"
        "def cost_tilted(params):\n"
        "    probs = qaoa_circuit(params, p=p)\n"
        "    loss = tilted_loss(probs, gamma_fixed)\n"
        "    # For comparison, we store the actual energy\n"
        "    history_tilted.append(standard_loss(probs))\n"
        "    return loss\n\n"
        "print('Optimizing Standard QAOA...')\n"
        "res_std = minimize(cost_std, init_params, method='COBYLA', options={'maxiter': 50})\n\n"
        "print('Optimizing Tilted QAOA...')\n"
        "res_tilted = minimize(cost_tilted, init_params, method='COBYLA', options={'maxiter': 50})\n\n"
        "plt.figure(figsize=(10, 5))\n"
        "plt.plot(history_std, label='Standard Loss')\n"
        "plt.plot(history_tilted, label=f'Tilted Loss (gamma={gamma_fixed})')\n"
        "plt.axhline(y=min(energies), color='r', linestyle='--', label='Ground State Energy')\n"
        "plt.xlabel('Iteration')\n"
        "plt.ylabel('Energy <H>')\n"
        "plt.title('QAOA: Standard vs Tilted Loss')\n"
        "plt.legend()\n"
        "plt.grid(True)\n"
        "plt.show()"
    ))

    # Task 2: HEA with Tilted Loss
    nb.cells.append(nbf.v4.new_markdown_cell("## Task 2: HEA with Tilted Loss"))
    nb.cells.append(nbf.v4.new_code_cell(
        "hea_layers = 2\n"
        "num_params = hea_layers * nodes * 2 + nodes\n"
        "init_params_hea = np.random.uniform(0, 2 * np.pi, num_params, requires_grad=True)\n\n"
        "history_hea_std = []\n"
        "history_hea_tilted = []\n\n"
        "def cost_hea_std(params):\n"
        "    probs = hea_circuit(params, layers=hea_layers)\n"
        "    loss = standard_loss(probs)\n"
        "    history_hea_std.append(loss)\n"
        "    return loss\n\n"
        "def cost_hea_tilted(params):\n"
        "    probs = hea_circuit(params, layers=hea_layers)\n"
        "    loss = tilted_loss(probs, gamma_fixed)\n"
        "    history_hea_tilted.append(standard_loss(probs))\n"
        "    return loss\n\n"
        "print('Optimizing HEA with Standard Loss...')\n"
        "minimize(cost_hea_std, init_params_hea, method='COBYLA', options={'maxiter': 100})\n\n"
        "print('Optimizing HEA with Tilted Loss...')\n"
        "minimize(cost_hea_tilted, init_params_hea, method='COBYLA', options={'maxiter': 100})\n\n"
        "plt.figure(figsize=(10, 5))\n"
        "plt.plot(history_hea_std, label='HEA Standard')\n"
        "plt.plot(history_hea_tilted, label=f'HEA Tilted (gamma={gamma_fixed})')\n"
        "plt.axhline(y=min(energies), color='r', linestyle='--', label='GS Energy')\n"
        "plt.xlabel('Iteration')\n"
        "plt.ylabel('Energy <H>')\n"
        "plt.title('HEA: Standard vs Tilted Loss')\n"
        "plt.legend()\n"
        "plt.grid(True)\n"
        "plt.show()"
    ))

    # Task 3: Different Gamma Values
    nb.cells.append(nbf.v4.new_markdown_cell("## Task 3: Effect of Different Fixed Gamma Values\n\n"
                                             "We examine and compare the algorithm performance using different fixed values of $\gamma$: 0.1, 5, 10, 20, 100."))
    
    nb.cells.append(nbf.v4.new_code_cell(
        "gamma_values = [0.1, 5, 10, 20, 100]\n"
        "results_gamma = {}\n\n"
        "for g in gamma_values:\n"
        "    print(f'Optimizing Tilted QAOA with gamma={g}...')\n"
        "    hist = []\n"
        "    def cost_tilted_g(params):\n"
        "        probs = qaoa_circuit(params, p=p)\n"
        "        loss = tilted_loss(probs, g)\n"
        "        hist.append(standard_loss(probs))\n"
        "        return loss\n"
        "    minimize(cost_tilted_g, init_params, method='COBYLA', options={'maxiter': 50})\n"
        "    results_gamma[g] = hist\n\n"
        "plt.figure(figsize=(10, 5))\n"
        "plt.plot(history_std, label='Standard Loss', linestyle='--', color='black', linewidth=2)\n"
        "for g in gamma_values:\n"
        "    plt.plot(results_gamma[g], label=f'gamma={g}')\n"
        "plt.axhline(y=min(energies), color='r', linestyle=':', label='Ground State Energy')\n"
        "plt.xlabel('Iteration')\n"
        "plt.ylabel('Energy <H>')\n"
        "plt.title('QAOA: Tilted Loss with Different Gamma Values')\n"
        "plt.legend()\n"
        "plt.grid(True)\n"
        "plt.show()"
    ))

    # Task 4: Gamma Parameter Tuning (Schedules)
    nb.cells.append(nbf.v4.new_markdown_cell("## Task 4: Evaluating Gamma Schedules\n\n"
                                             "We explore how changing $\gamma$ during optimization affects performance.\n"
                                             "The schedules tested are:\n"
                                             "1. **Linear**: $\gamma(t) = \gamma_0 + (\gamma_f - \gamma_0) \\frac{t}{T}$\n"
                                             "2. **Sigmoid**: Smooth transition\n"
                                             "3. **Exponential**: Growth factor base\n"
                                             "4. **Logarithmic**: Slower growth"))
    
    nb.cells.append(nbf.v4.new_code_cell(
        "T = 50 # Total iterations\n"
        "gamma_0, gamma_f = 0.01, 2.0\n\n"
        "def get_gamma_schedule(name, t, T):\n"
        "    if name == 'linear':\n"
        "        return gamma_0 + (gamma_f - gamma_0) * (t / T)\n"
        "    elif name == 'sigmoid':\n"
        "        return gamma_0 + (gamma_f - gamma_0) / (1 + np.exp(-10 * (t/T - 0.5)))\n"
        "    elif name == 'exponential':\n"
        "        return gamma_0 * (gamma_f / gamma_0)**(t / T)\n"
        "    elif name == 'logarithmic':\n"
        "        return gamma_0 + (gamma_f - gamma_0) * np.log(1 + t) / np.log(1 + T)\n"
        "    return gamma_0\n\n"
        "schedules = ['linear', 'sigmoid', 'exponential', 'logarithmic']\n"
        "results_schedules = {}\n\n"
        "for sched in schedules:\n"
        "    print(f'Testing {sched} schedule...')\n"
        "    params = init_params.copy()\n"
        "    hist = []\n"
        "    \n"
        "    # Custom simple gradient descent or optimization loop to handle dynamic gamma\n"
        "    for t in range(T):\n"
        "        current_gamma = get_gamma_schedule(sched, t, T)\n"
        "        \n"
        "        def local_cost(opt_params):\n"
        "            pr = qaoa_circuit(opt_params, p=p)\n"
        "            return tilted_loss(pr, current_gamma)\n"
        "        \n"
        "        # Perform 1 step of optimization per schedule step (or use a minimizer per step)\n"
        "        res = minimize(local_cost, params, method='COBYLA', options={'maxiter': 5})\n"
        "        params = res.x\n"
        "        # Log the actual expectation value\n"
        "        hist.append(standard_loss(qaoa_circuit(params, p=p)))\n"
        "    \n"
        "    results_schedules[sched] = hist\n\n"
        "plt.figure(figsize=(12, 6))\n"
        "for name, data in results_schedules.items():\n"
        "    plt.plot(data, label=name)\n"
        "plt.axhline(y=min(energies), color='black', linestyle=':', label='Optimal')\n"
        "plt.xlabel('Optimization Step')\n"
        "plt.ylabel('Energy')\n"
        "plt.title('Effect of Gamma Schedules on QAOA Performance')\n"
        "plt.legend()\n"
        "plt.grid(alpha=0.3)\n"
        "plt.show()"
    ))

    # Save the notebook
    path = os.path.abspath('qaoa_tilted_loss_analysis.ipynb')
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    return path

if __name__ == '__main__':
    notebook_path = generate_notebook()
    print(f'Notebook generated at: {notebook_path}')
