import matplotlib.pyplot as plt
import os
import sys

def read_file(name):
    N = []
    M = []
    S = []
    filepath = os.path.join(sys.path[0], name)
    with open(filepath, 'r') as input:
        for line in input:
            n, m, sigma, _ = line.split()
            N.append(n)
            M.append(float(m))
            S.append(float(sigma))
    return N, M, S

plt.title('Precision analysis')
N, M, S = read_file('data/precision_uniform_weights.txt')
plt.plot(N, S, label = 'uniformly distributed weights')
N, M, S = read_file('data/precision_exponential_weights_scale_1.txt')
plt.plot(N, S, label = 'exponentialy distributed weights')
plt.yscale('log')
plt.legend()
plt.xlabel('Number of binary variables')
plt.ylabel('Standard deviation of argmin')
plt.savefig('plots/precision_analysis.png')
plt.show()