from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA, NumPyEigensolver
from qiskit_finance.applications.optimization.portfolio_optimization import PortfolioOptimization
from qiskit_optimization.converters import LinearEqualityToPenalty
from qiskit import Aer
import numpy as np
import matplotlib.pyplot as plt




