import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    dim = spk_pop.shape[1]
    
    # Select the best, second best, and worst sparks
    sorted_indices = np.argsort(spk_fit)
    best_idx, second_best_idx, worst_idx = sorted_indices[0], sorted_indices[1], sorted_indices[-1]
    
    # Calculate difference vectors
    diff_vector1 = spk_pop[best_idx] - spk_pop[worst_idx]
    diff_vector2 = spk_pop[second_best_idx] - idv
    
    # Generate random scaling factors
    F1 = np.random.uniform(0.5, 1.5, dim)
    F2 = np.random.uniform(0.2, 0.8, dim)
    
    # Apply differential mutation
    mutant = idv + F1 * diff_vector1 + F2 * diff_vector2
    
    # Add Lévy flight for better exploration
    levy = np.random.standard_cauchy(dim)
    mutant += 0.02 * levy * (self.ub - self.lb)
    
    # Apply adaptive polynomial mutation
    p_m = max(0.05, min(0.2, 1 - (self.cur_gen / self.max_gen)**1.5))
    eta_m = 20 + 10 * np.sqrt(self.cur_gen / self.max_gen)
    for j in range(dim):
        if np.random.random() < p_m:
            r = np.random.random()
            if r < 0.5:
                delta = (2 * r) ** (1 / (eta_m + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r)) ** (1 / (eta_m + 1))
            mutant[j] += delta * (self.ub - self.lb)
    
    # Apply opposition-based learning with adaptive probability
    p_obl = max(0.05, min(0.15, 1 - np.sqrt(self.cur_gen / self.max_gen)))
    if np.random.random() < p_obl:
        opposite = self.lb + self.ub - mutant
        mutant = np.where(np.random.random(dim) < 0.5, mutant, opposite)
    
    # Apply local search with small probability
    if np.random.random() < 0.1:
        local_step = np.random.uniform(-0.01, 0.01, dim) * (self.ub - self.lb)
        mutant += local_step
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.05, dim)
    mutant += noise * (self.ub - self.lb)
    
    # Ensure the mutant is within bounds
    mutation_spark = self.remap(mutant, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
