import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    sorted_idx = np.argsort(spk_fit)
    
    # Select the best performers
    best_idx = sorted_idx[:int(num_spk * self.gm_ratio)]
    best_centroid = np.mean(spk_pop[best_idx, :], axis=0)
    
    # Compute differential vectors
    diff_vectors = spk_pop[best_idx] - idv
    diff_vector = np.mean(diff_vectors, axis=0)
    
    # Add scaled differential vector, Gaussian noise, Levy flight
    scale_factor = np.random.uniform(0.4, 0.9, idv.shape)
    noise = np.random.normal(0, np.abs(diff_vector), idv.shape)
    
    # Levy flight perturbation
    beta = 1.5
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, idv.shape)
    v = np.random.normal(0, 1, idv.shape)
    levy_perturbation = u / np.abs(v) ** (1 / beta)
    
    mutation_spark = self.remap(idv + scale_factor * diff_vector + noise + 0.01 * levy_perturbation, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
