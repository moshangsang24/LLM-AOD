import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    dim = spk_pop.shape[1]
    
    # Select the best, second best, and worst sparks
    sorted_indices = np.argsort(spk_fit)
    best_idx, second_best_idx, worst_idx = sorted_indices[0], sorted_indices[1], sorted_indices[-1]
    
    # Calculate difference vectors
    diff1 = spk_pop[best_idx] - spk_pop[worst_idx]
    diff2 = spk_pop[second_best_idx] - idv
    
    # Adaptive scaling factors
    F1 = np.random.beta(2, 2) * 1.2  # Beta distribution for more controlled randomness
    F2 = np.random.beta(2, 2) * 0.8
    
    # Apply differential mutation with two difference vectors
    mutant = idv + F1 * diff1 + F2 * diff2
    
    # Levy flight for occasional large jumps
    levy = np.random.standard_cauchy(dim) * 0.1
    mutant += levy
    
    # Adaptive polynomial mutation
    pm_prob = 0.3 * (1 - self.cur_gen / self.max_gen)  # Decreasing probability over generations
    eta = 20 + 5 * (self.cur_gen / self.max_gen)  # Increasing distribution index
    for j in range(dim):
        if np.random.random() < pm_prob:
            rand = np.random.random()
            if rand < 0.5:
                delta = (2 * rand) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - rand)) ** (1 / (eta + 1))
            mutant[j] += delta * (self.ub - self.lb) * 0.1
    
    # Ensure the mutant is within bounds
    mutation_spark = self.remap(mutant, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
