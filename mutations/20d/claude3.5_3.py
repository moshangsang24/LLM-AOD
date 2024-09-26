import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    dim = spk_pop.shape[1]
    
    # Select the best, second best, and worst sparks
    sorted_indices = np.argsort(spk_fit)
    best_idx, second_best_idx, median_idx, worst_idx = sorted_indices[0], sorted_indices[1], sorted_indices[num_spk//2], sorted_indices[-1]
    
    # Calculate difference vectors
    diff1 = spk_pop[best_idx] - spk_pop[worst_idx]
    diff2 = spk_pop[second_best_idx] - spk_pop[median_idx]
    diff3 = idv - spk_pop[best_idx]
    
    # Adaptive scaling factors
    F1 = np.random.beta(2, 2) * 1.2
    F2 = np.random.beta(2, 2) * 0.8
    F3 = np.random.beta(2, 2) * 0.5
    
    # Apply differential mutation with three difference vectors
    mutant = idv + F1 * diff1 + F2 * diff2 + F3 * diff3
    
    # Levy flight for occasional large jumps
    levy = np.random.standard_cauchy(dim) * 0.05 * (1 - self.cur_gen / self.max_gen)
    mutant += levy
    
    # Adaptive polynomial mutation
    pm_prob = max(0.1, 0.4 * (1 - self.cur_gen / self.max_gen))
    eta = 15 + 10 * (self.cur_gen / self.max_gen)
    for j in range(dim):
        if np.random.random() < pm_prob:
            rand = np.random.random()
            if rand < 0.5:
                delta = (2 * rand) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - rand)) ** (1 / (eta + 1))
            mutant[j] += delta * (self.ub - self.lb) * 0.1
    
    # Occasional reset of dimensions
    reset_prob = 0.03 * (1 - self.cur_gen / self.max_gen)
    reset_mask = np.random.random(dim) < reset_prob
    mutant[reset_mask] = np.random.uniform(self.lb, self.ub, np.sum(reset_mask))
    
    # Local search around the best solution
    if np.random.random() < 0.1:
        local_step = np.random.normal(0, 0.01, dim)
        mutant = spk_pop[best_idx] + local_step * (self.ub - self.lb)
    
    # Ensure the mutant is within bounds
    mutation_spark = self.remap(mutant, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
