import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    dim = spk_pop.shape[1]
    
    # Select the best and worst sparks
    best_idx = np.argmin(spk_fit)
    worst_idx = np.argmax(spk_fit)
    
    # Calculate the difference vector
    diff_vector = spk_pop[best_idx] - spk_pop[worst_idx]
    
    # Generate random scaling factors
    F = np.random.uniform(0.4, 1.6, dim)
    
    # Apply differential mutation
    mutant = idv + F * diff_vector
    
    # Add Lévy flight for better exploration
    levy = np.random.standard_cauchy(dim)
    mutant += 0.01 * levy * (self.ub - self.lb)
    
    # Apply polynomial mutation with adaptive probability
    p_m = 0.1 + 0.1 * (1 - self.cur_gen / self.max_gen)
    eta_m = 20 + 10 * (self.cur_gen / self.max_gen)
    for j in range(dim):
        if np.random.random() < p_m:
            r = np.random.random()
            if r < 0.5:
                delta = (2 * r) ** (1 / (eta_m + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r)) ** (1 / (eta_m + 1))
            mutant[j] += delta * (self.ub - self.lb)
    
    # Apply opposition-based learning with probability
    if np.random.random() < 0.1:
        opposite = self.lb + self.ub - mutant
        mutant = np.where(np.random.random(dim) < 0.5, mutant, opposite)
    
    # Ensure the mutant is within bounds
    mutation_spark = self.remap(mutant, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
