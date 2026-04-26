from numpy import array, identity
from numpy import log, exp, cos, sin, conj, transpose, sqrt
from numpy import random, pi, linspace
from numpy import trace, real
from numpy import kron, eye, diag
from numpy import linalg
from qiskit.quantum_info import random_density_matrix, random_statevector

import matplotlib.pyplot as plt

eps = 1e-12
def I():
    return identity(2)
def X():
    return array([[0, 1], [1, 0]])
def Y():
    return array([[0, -1j], [1j, 0]])
def Z():
    return array([[1, 0], [0, -1]])
def Rx(theta):
    return cos(theta / 2) * I() - 1j * sin(theta / 2) * X()
def Ry(theta):
    return cos(theta / 2) * I() - 1j * sin(theta / 2) * Y()
def Rz(theta):
    return cos(theta / 2) * I() - 1j * sin(theta / 2) * Z()
def Nkron(matrices):
    dim = len(matrices)
    if dim == 1:
        kronMatrix = matrices[0]
    else:
        kronMatrix = identity(1)
        for i in range(dim):
            kronMatrix = kron(kronMatrix, matrices[i])
    return kronMatrix

def RQ(HTheta, t, rho, dim):
    if linalg.norm(rho - identity(2 ** dim)) < eps:
        return real(1 / t * log(trace(exp(t * HTheta))))
    else:
        return real(1 / t * log(trace(exp(t * HTheta + log(rho)))))

# System size
dim = 3
# Rho
# rho = Nkron([Rz(random.random()) for _ in range(dim)])
# HTheta = random_density_matrix(2 ** dim).data.reshape(2 ** dim, 2 ** dim)
rho = random_density_matrix(2 ** dim).data.reshape(2 ** dim, 2 ** dim)

# rho = Nkron([I() for _ in range(dim)])
# rho = random_statevector(2 ** dim).data.reshape(2 ** dim, 1)
# rho = rho @ transpose(conj(rho))
print(rho)
# t
# T = linspace(start=-10, stop=10, num=10)
# T = [i * 0.5 for i in range(-5, 7, 2)] + [0]
T = list(range(-10, 12, 4)) + [0]
# T = [i * 0.2 for i in range(-5, 7, 2)] + [0]
# T = range(-10, 12)

# 1. Set rho to be identity
# Hamiltonian, range theta
# Theta = linspace(start=0, stop=1, num=50) * 2 * pi
Theta = linspace(start=0, stop=1, num=50) * pi / 2 + 3 * pi / 4

colors = ['red', 'blue', 'green']
for t in T:
    if t == 0:
        RQList = []
        for theta in Theta:
            HTheta = Nkron([Rx(theta) @ Rz(theta) for _ in range(dim)])
            # HTheta = Nkron([Rx(theta) for _ in range(dim)])
            RQList.append(real(trace(HTheta @ rho)))
        # plt.plot(Theta, RQList, label=r't = 0', color='green')
        plt.plot(Theta, RQList, label=r't = 0')
        # plt.scatter(Theta, RQList)
    else:
        RQList = []
        for theta in Theta:
            HTheta = Nkron([Rx(theta) @ Rz(theta) for _ in range(dim)])
            # HTheta = Nkron([Rx(theta) for _ in range(dim)])
            RQList.append(RQ(HTheta, t, rho, dim))
        if t < 0:
            color = colors[0]
        else:
            color = colors[1]
        # plt.plot(Theta, RQList, label=r't = ' + str(t), color=color)
        plt.plot(Theta, RQList, label=r't = ' + str(t))
        # plt.scatter(Theta, RQList)
plt.xlabel("theta")
plt.ylabel("RQ")
plt.legend()
plt.show()



# Expression
