import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    sorted_idx = np.argsort(spk_fit)
    
    # Select the top and bottom performers
    top_idx = sorted_idx[:int(num_spk * self.gm_ratio)]
    bottom_idx = sorted_idx[-int(num_spk * self.gm_ratio):]
    
    # Compute centroids for top and bottom performers
    top_centroid = np.mean(spk_pop[top_idx, :], axis=0)
    bottom_centroid = np.mean(spk_pop[bottom_idx, :], axis=0)
    
    # Differential vector between top and bottom centroids
    diff_vector = top_centroid - bottom_centroid
    
    # Add scaled differential vector, Gaussian noise, Levy flight, and Gaussian mutation perturbation
    scale_factor = np.random.uniform(0.4, 0.9, idv.shape)
    noise = np.random.normal(0, 0.1, idv.shape)
    
    # Levy flight perturbation
    beta = 1.5
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, idv.shape)
    v = np.random.normal(0, 1, idv.shape)
    levy_perturbation = u / np.abs(v) ** (1 / beta)
    
    # Gaussian mutation perturbation for increased diversity
    mutation_perturbation = np.random.normal(0, 1, idv.shape) * 0.1
    
    mutation_spark = self.remap(idv + scale_factor * diff_vector + noise + 0.01 * levy_perturbation + mutation_perturbation, self.lb, self.ub).reshape(1, -1)

    return mutation_spark
