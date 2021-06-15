import random as rd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import config

from azure.quantum.optimization import Problem, ProblemType, Term
from azure.quantum.optimization import ParallelTempering, SimulatedAnnealing, Tabu, QuantumMonteCarlo
from azure.quantum import Workspace
from numpy.polynomial.polynomial import Polynomial, polyval

def create_workspace():
    workspace = Workspace(
        subscription_id = config._SUBSCRIPTION_ID,  
        resource_group = config._RESOURCE_GROUP,
        name = config._NAME,    
        location = config._LOCATION
    )
    workspace.login()
    return workspace

def product(subset, W):
    result = 1
    for i in subset:
        result *= W[i]
    return result

def reduce_subset(subset, problem_type):
    ### Simplify term when a subset contains the same variable more than once.
    reduced_variable_subset = []
    
    ### For an Ising problem, keep a variable iff its multiplicity is odd.
    if problem_type == ProblemType.ising:
        for i in subset:
            if i not in reduced_variable_subset:
                reduced_variable_subset.append(i)
            else:
                reduced_variable_subset.remove(i)
    
    ### For a pubo problem, keep a variable iff its multiplicity is non zero.
    if problem_type == ProblemType.pubo:
        for i in subset:
            if i not in reduced_variable_subset:
                reduced_variable_subset.append(i)
                
    return set(reduced_variable_subset)
             
def create_problem(cost_function, nb_binary_variables) -> Problem:
    ### the cost_function is given as a list of polynomial coefficients.
    
    problem_type = ProblemType.ising

    indices = range(nb_binary_variables)
    random_weights = np.array([rd.random() for _ in indices])
    # random_weights = np.array([np.random.exponential(scale=100) for _ in indices])
    random_weights = random_weights/sum(random_weights)               ### Normalize random_weights to sum to 1.

    reduced_variable_subset_list = []
    weight_list = []
    for degree, coefficient in enumerate(cost_function):
        for variable_subset_of_size_degree in itertools.product(indices, repeat=degree):
            weight = coefficient*product(variable_subset_of_size_degree, random_weights)
            reduced_variable_subset = reduce_subset(variable_subset_of_size_degree, problem_type)
            if reduced_variable_subset not in reduced_variable_subset_list: 
                reduced_variable_subset_list.append(reduced_variable_subset)
                weight_list.append(weight)
            else:
                i = reduced_variable_subset_list.index(reduced_variable_subset)
                weight_list[i] += weight

    terms = []      
    for weight, reduced_variable_subset in zip(weight_list, reduced_variable_subset_list):
        terms.append(
            Term(
                c = weight,
                indices = list(reduced_variable_subset)
            )
        )
     
    return random_weights, Problem(name="Continuous cost function", problem_type=problem_type, terms=terms)

def get_continuous_variable_result(result, random_weights, Polynomial):
    x_min = 0
    for i, w in zip(result['configuration'], random_weights):
        b = result['configuration'][i]
        x_min += w*b
    
    cost = 0
    for d, p in enumerate(Polynomial):
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
    # plt.savefig('plots/cost_function_and_found_minimum.png')
    plt.show()

def main():
    Polynomial = [0, 0, -0.2, 0.06, 0.3]      ### P = -0.2*x**2 + 0.06*x**3 + 0.3*x**4
    n = 10                                    ### Number of binary variables to discretize x.

    random_weights, problem = create_problem(Polynomial, n)
    workspace = create_workspace()
    solver = QuantumMonteCarlo(workspace)
    result = solver.optimize(problem)

    x_min, cost = get_continuous_variable_result(result, random_weights, Polynomial)
    print('\n x_min =', x_min)
    print('cost =', cost)

    plot_cost_function_and_result(Polynomial, x_min)


if __name__ == "__main__":
    main()


    
    









