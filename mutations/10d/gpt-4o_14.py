import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    sorted_idx = np.argsort(spk_fit)
    
    # Select the top performers for exploitation
    top_idx = sorted_idx[:int(num_spk * self.gm_ratio)]
    top_centroid = np.mean(spk_pop[top_idx, :], axis=0)
    
    # Select the bottom performers for diversity
    bottom_idx = sorted_idx[-int(num_spk * self.gm_ratio):]
    bottom_centroid = np.mean(spk_pop[bottom_idx, :], axis=0)
    
    # Compute differential vectors
    diff_vector_top = top_centroid - idv
    diff_vector_bottom = idv - bottom_centroid
    
    # Combine differential vectors for exploitation and diversity
    combined_diff_vector = diff_vector_top + diff_vector_bottom
    
    # Adaptive scaling factor based on performance
    scale_factor = np.random.uniform(0.5, 1.0, idv.shape) * (1 - spk_fit[top_idx[0]] / (spk_fit[bottom_idx[-1]] + EPS))
    
    # Gaussian noise and random uniform perturbation
    noise = np.random.normal(0, np.abs(combined_diff_vector), idv.shape)
    perturbation = np.random.uniform(-0.1, 0.1, idv.shape)
    
    mutation_spark = self.remap(idv + scale_factor * combined_diff_vector + noise + perturbation, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
