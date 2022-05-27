from typing import Callable
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt

class BRKGA:
    def __init__(self
                , shape:tuple
                , n_ind:int
                , breakdown:float
                , bias:float
                , ngens:int
                , fitness_function:Callable
                , initial_population:np.ndarray=None
        ):
        self.shape = shape
        self.n_ind = n_ind
        self.perc_elite, self.perc_cross, self.perc_random = breakdown #0.3, 0.4, 0.3
        if initial_population is None:
            self.population = get_initial_population(shape, self.n_ind)
        else:
            self.population = initial_population
        self.bias = bias
        self.ngens = ngens
        self.fitness = np.zeros(ngens, dtype=float)
        self.fitness_function = fitness_function
        base_fitness = self.fitness_function(self.population)
        self.population = self.population[np.argsort(base_fitness)]
        
        self.curr_fitness = np.min(base_fitness)
        self.fitness[0] = self.curr_fitness
        #print(f'Generation: 0, fitness : {self.curr_fitness : .2f}')
    
    def begin_evolving(self)->None:
        for i in range(self.ngens):
            self.evolve_population()
            current_gen_fitness = self.fitness_function(self.population)
            self.population = self.population[np.argsort(current_gen_fitness)]
            self.curr_fitness = np.min(current_gen_fitness)
            self.fitness[i] = self.curr_fitness
            #print(f'Generation: {i}, fitness : {self.curr_fitness : .2f}')
        return self
    
    def evolve_population(self)->None:
        n_elite = int(self.perc_elite*self.population.shape[0])
        n_cross = int(self.perc_cross*self.population.shape[0])
        elite = self.population[:n_elite]
        non_elite = self.population[n_elite:]
        elite_cross = elite[np.random.randint(0, n_elite, n_cross)]
        non_elite_cross = non_elite[np.random.randint(0, n_cross, n_cross)]
        picks = (np.random.random(non_elite_cross.shape)>(1-self.bias))
        cross = non_elite_cross
        cross[picks] = elite_cross[picks]
        self.population[n_elite:n_elite + n_cross] = cross
        self.population[n_elite + n_cross:] = get_initial_population(self.shape, self.population.shape[0] - n_elite - n_cross)
        
    def plot_evolution(self)->None:
        pd.Series(
            self.fitness
        ).plot()
        plt.ylabel('Fitness')
        plt.xlabel('# generations')

import copy
class ParallelBRKGA(BRKGA):
    def __init__(self
                , shape:tuple
                , n_ind:int
                , breakdown:float
                , bias:float
                , ngens:int
                , fitness_function:Callable
                , npopulations:int
                , initial_population:np.ndarray=None
        ):
        self.shape = shape
        self.n_ind = n_ind
        self.breakdown = breakdown #0.3, 0.4, 0.3
        self.bias = bias
        self.ngens = ngens
        self.fitness_function = fitness_function
        self.brkgas = [copy.deepcopy(
            BRKGA(
                shape=shape
                , n_ind=  n_ind
                , breakdown = breakdown
                , bias = bias
                , ngens = ngens
                , fitness_function = fitness_function
                , initial_population= initial_population
            )
        ) for _ in range(npopulations)]
    
    def begin_evolving(self
                    , n_jobs:int = 8
                    , pre_dispatch:str="2*n_jobs"
                    , backend:str="loky"
        ) -> None:
        parallel = Parallel(
                n_jobs=n_jobs, pre_dispatch=pre_dispatch, backend=backend, max_nbytes='50M'
        )
        self.brkgas = parallel(
                    delayed(brkga.begin_evolving)() 
                    for brkga in self.brkgas
        )
    
    def begin_elitist_mating(self
                            , n_ind_per_instance:int
                            , ngens=int
                            ):
        new_population = np.vstack(
            [brkga.population[:n_ind_per_instance] for brkga in self.brkgas]
        )
        self.elitist_brkga = BRKGA(
                shape=self.shape
                , n_ind= new_population.shape[0]#self.n_ind
                , breakdown = self.breakdown
                , bias = self.bias
                , ngens = self.ngens if ngens is None else ngens
                , fitness_function = self.fitness_function
                , initial_population = new_population
            )
        self.elitist_brkga.begin_evolving()
    
    def plot_evolution(self):
        pd.DataFrame(
            np.array(
                list(map(lambda x: x.fitness, self.brkgas))
            ).T
        ).plot()
        plt.ylabel('Fitness')
        plt.xlabel('# generations')
    
def get_initial_population(shape, n_ind):
    return np.random.random((n_ind, *shape))