
import numpy as np
# from scipy.spatial.distance import cdist
# from brkga import BRKGA, ParallelBRKGA
import matplotlib.pyplot as plt

def instance(npoints, max_x=200, max_y=200):
    xs = np.random.randint(
        0,max_x, npoints
    )
    ys = np.random.randint(
        0,max_y, npoints
    )
    return np.vstack((xs,ys)).T

def decode_pop(population):
    decoded_pop = np.zeros(population[:,:,0].shape, dtype=int)
    decoded_pop[:,1:] = np.argsort(population[:,1:,0], axis=1)
    decoded_pop += 1
    decoded_pop[:,0] = 0
    return decoded_pop

def evaluate_pop(distance_matrix, population):
    decoded_pop = decode_pop(population)
    return np.sum(
        distance_matrix[decoded_pop[:,:-1], decoded_pop[:,1:]]
        , axis=1
    ) + distance_matrix[decoded_pop[:,-1], 0]

def plot_solution(instance, solution):
    plot_solution = np.hstack([solution,[0]])
    plt.plot(instance[:,0][plot_solution], instance[:,1][plot_solution], '-.o')
    plt.plot(instance[0,0], instance[0,1], '-.o')
