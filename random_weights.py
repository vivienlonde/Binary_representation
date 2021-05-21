import random as rd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import config

from azure.quantum.optimization import Problem, ProblemType, Term
from azure.quantum.optimization import ParallelTempering, SimulatedAnnealing, Tabu, QuantumMonteCarlo
from azure.quantum import Workspace
from numpy.polynomial.polynomial import polyval

# Workspace information
workspace = Workspace(
    subscription_id = config._SUBSCRIPTION_ID,  
    resource_group = config._RESOURCE_GROUP,
    name = config._NAME,    
    location = config._LOCATION
)
workspace.login()

def product(subset, W):
    result = 1
    for i in subset:
        result *= W[i]
    return result

def reduce_subset(subset, problem_type):
    ### Simplify term when a subset contains the same variable more than once.
    reduced_subset = []
    
    ### For an Ising problem, keep a variable iff its multiplicity is odd.
    if problem_type == ProblemType.ising:
        for i in subset:
            if i not in reduced_subset:
                reduced_subset.append(i)
            else:
                reduced_subset.remove(i)
    
    ### For a pubo problem, keep a variable iff its multiplicity is non zero.
    if problem_type == ProblemType.pubo:
        for i in subset:
            if i not in reduced_subset:
                reduced_subset.append(i)
                
    return reduced_subset
                
def create_problem(cost_function, nb_binary_variables) -> Problem:
    ### the cost_function is given as a list of polynomial coefficients.
    
    problem_type = ProblemType.ising

    indices = range(nb_binary_variables)
    W = np.array([rd.random() for _ in indices])
    W = W/sum(W)                                  ### normalize W to sum to 1.
    # print('W:', W)

    terms = []
    for degree, coefficient in enumerate(cost_function):
        for subset_of_size_d in itertools.product(indices, repeat=degree):
            weight = coefficient*product(subset_of_size_d, W)
            reduced_subset = reduce_subset(subset_of_size_d, problem_type)
            terms.append(
                Term(
                    c = weight,
                    indices = reduced_subset
                )
            )
    
    return W, Problem(name="Continuous cost function", problem_type=problem_type, terms=terms)

def get_continuous_variable_result(result):
    x_min = 0
    for i, w in zip(result['configuration'], W):
        b = result['configuration'][i]
        x_min += w*b
    
    cost = 0
    for d, p in enumerate(P):
        cost += p*x_min**d
    
    return x_min, cost

def plot_cost_function_and_result(P, x_min):
    nb_plot_points = 100
    a = -1
    b = 1
    X_plot = np.linspace(a, b, nb_plot_points) 
    Y_P_plot = [polyval(x, P) for x in X_plot]
    plt.plot(X_plot, Y_P_plot, label = 'Cost function')
    plt.plot([x_min], [polyval(x_min, P)], 'ro')
    plt.legend()
    plt.savefig('cost_function_and_found_minimum.png')
    plt.show()


P = [0, 0, -0.2, 0.06, 0.3]      ### P = -0.35/2*x**2 + 0.2/3*x**3 + 1/4*x**4
n = 10                           ### number of binary variables to discretize x.

W, problem = create_problem(P, n)
solver = SimulatedAnnealing(workspace)
result = solver.optimize(problem)

x_min, cost = get_continuous_variable_result(result)
print('\n x_min =', x_min)
print('cost =', cost)

plot_cost_function_and_result(P, x_min)
    
    









