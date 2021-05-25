from random_weights_discretization import *
import os
import sys

Polynomial = [0, 0, -0.2, 0.06, 0.3]      ### P = -0.2*x**2 + 0.06*x**3 + 0.3*x**4
nb_trials = 20

# filepath = os.path.join(sys.path[0], 'data/precision_uniform_weights.txt')
# filepath = os.path.join(sys.path[0], 'data/precision_exponential_weights_scale_1.txt')
filepath = os.path.join(sys.path[0], 'data/precision_exponential_weights_scale_100.txt')
with open(filepath, 'w') as output:
    
    for n in range(2, 21):
        print('n:', n) ### Number of binary variables to discretize x.

        x_min_list = []
        for trial in range(nb_trials):
            print('trial:', trial)
            random_weights, problem = create_problem(Polynomial, n)
            workspace = create_workspace()
            solver = SimulatedAnnealing(workspace)
            result = solver.optimize(problem)

            x_min, _ = get_continuous_variable_result(result, random_weights, Polynomial)
            x_min_list.append(x_min)

        output.write(str(n) + ' ' + str(np.average(x_min_list)) + ' ' + str(np.std(x_min_list))+ ' ' + str(nb_trials) + '\n')
        


