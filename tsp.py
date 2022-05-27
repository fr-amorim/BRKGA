
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

# if __name__=='__main__':
#     import matplotlib.pyplot as plt
#     from functools import partial
#     points = instance(20)
#     distance_matrix = cdist(points, points)
    
#     initial_solution=np.arange(points.shape[0])


#     fitness_function  = partial(evaluate_pop, distance_matrix)
#     brkga = ParallelBRKGA(
#         shape=(points.shape[0],1)
#         , n_ind=1000
#         , breakdown=[0.3,0.4,0.3]
#         , bias=0.7
#         , ngens=200
#         , fitness_function = fitness_function
#         , initial_population=None
#         , npopulations=8
#     )
    
#     brkga.begin_evolving()
    
#     brkga.begin_elitist_mating(n_ind_per_instance=10, ngens=200)
    
    
    
#     final_solution = decode_pop(brkga.elitist_brkga.population)[0]

#     plt.plot(points[:,0][initial_solution], points[:,1][initial_solution], '-.o')
#     plt.plot(points[:,0][final_solution], points[:,1][final_solution], '--o')
#     plt.show()

#     print(
#     np.sum(
#             distance_matrix[final_solution[:-1], final_solution[1:]]
#         )
#     ,
#     np.sum(
#             distance_matrix[initial_solution[:-1], initial_solution[1:]]
#         )
#     )
    
    
#     ##brute force attempt in as many iterations
#     random_population = np.random.random((brkga.n_ind * len(brkga.brkgas) + brkga.elitist_brkga.population.shape[0], *brkga.shape))
#     print(
#         np.min(
#             evaluate_pop(distance_matrix, random_population)
#         )
#     )
    
#     # print('elitist mating')
#     # next_population = np.vstack(
#     #     [x.population[:10] for x in out]
#     # )
    
#     # brkga = BRKGA(
#     #     shape=(points.shape[0],1)
#     #     , n_ind=next_population.shape[0]
#     #     , breakdown=[0.2,0.3,0.5]
#     #     , bias=0.7
#     #     , ngens=500
#     #     , fitness_function = fitness_function
#     #     , initial_population= next_population
#     # )
    
#     # brkga.begin_evolving()
#     # final_solution = decode_pop(brkga.population)[0]
    
#     # print(np.sum(
#     #         distance_matrix[final_solution[:-1], final_solution[1:]]
#     #     ))