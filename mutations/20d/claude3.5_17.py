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
    diff3 = np.mean(spk_pop[sorted_indices[:5]], axis=0) - idv
    
    # Adaptive scaling factors
    F1 = np.random.beta(2, 2) * 1.2
    F2 = np.random.beta(2, 2) * 0.8
    F3 = np.random.beta(2, 2) * 0.5
    
    # Apply differential mutation with three difference vectors
    mutant = idv + F1 * diff1 + F2 * diff2 + F3 * diff3
    
    # Levy flight for occasional large jumps
    levy = np.random.standard_cauchy(dim) * 0.05 * np.exp(-0.02 * self.cur_gen)
    mutant += levy
    
    # Adaptive polynomial mutation
    pm_prob = max(0.1, 0.3 * np.exp(-0.005 * self.cur_gen))
    eta = 20 + 15 * (1 - np.exp(-0.01 * self.cur_gen))
    for j in range(dim):
        if np.random.random() < pm_prob:
            rand = np.random.random()
            if rand < 0.5:
                delta = (2 * rand) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - rand)) ** (1 / (eta + 1))
            mutant[j] += delta * (self.ub - self.lb) * 0.1
    
    # Occasional reset of dimensions
    reset_prob = 0.02 * np.exp(-0.015 * self.cur_gen)
    reset_mask = np.random.random(dim) < reset_prob
    mutant[reset_mask] = np.random.uniform(self.lb, self.ub, np.sum(reset_mask))
    
    # Local search around the best solution
    if np.random.random() < 0.2:
        local_step = np.random.normal(0, 0.01 * np.exp(-0.005 * self.cur_gen), dim)
        mutant = spk_pop[best_idx] + local_step * (self.ub - self.lb)
    
    # Crossover with the best solution
    cr = 0.8 * (1 - 0.3 * self.cur_gen / self.max_gen)
    mask = np.random.random(dim) < cr
    mutant = mask * mutant + (1 - mask) * spk_pop[best_idx]
    
    # Adaptive neighborhood search
    if np.random.random() < 0.25:
        neighbors = np.random.choice(num_spk, 3, replace=False)
        neighborhood_center = np.mean(spk_pop[neighbors], axis=0)
        mutant = neighborhood_center + np.random.normal(0, 0.1, dim) * (self.ub - self.lb)
    
    # Opposition-based learning
    if np.random.random() < 0.1:
        opposite = self.lb + self.ub - mutant
        if np.random.random() < 0.5:
            mutant = opposite
    
    # Gaussian mutation
    if np.random.random() < 0.15:
        gaussian_noise = np.random.normal(0, 0.1, dim)
        mutant += gaussian_noise * (self.ub - self.lb)
    
    # Simulated Binary Crossover (SBX)
    if np.random.random() < 0.2:
        eta_c = 20
        random_partner = spk_pop[np.random.randint(num_spk)]
        beta = np.random.random(dim)
        beta = np.where(beta <= 0.5, (2 * beta) ** (1 / (eta_c + 1)), (1 / (2 * (1 - beta))) ** (1 / (eta_c + 1)))
        mutant = 0.5 * ((1 + beta) * mutant + (1 - beta) * random_partner)
    
    # Adaptive mutation rate
    mutation_rate = 0.1 * (1 - self.cur_gen / self.max_gen)
    mutation_mask = np.random.random(dim) < mutation_rate
    mutant[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask)) * (self.ub - self.lb)
    
    # Ensure the mutant is within bounds
    mutation_spark = self.remap(mutant, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
