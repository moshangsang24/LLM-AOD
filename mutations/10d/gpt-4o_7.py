import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    sorted_idx = np.argsort(spk_fit)
    
    # Select the best and worst performers
    best_idx = sorted_idx[:int(num_spk * self.gm_ratio)]
    worst_idx = sorted_idx[-int(num_spk * self.gm_ratio):]
    
    # Compute centroids for best and worst performers
    best_centroid = np.mean(spk_pop[best_idx, :], axis=0)
    worst_centroid = np.mean(spk_pop[worst_idx, :], axis=0)
    
    # Differential vector between best and worst centroids
    diff_vector = best_centroid - worst_centroid
    
    # Add differential vector with a scaling factor and Gaussian noise
    scale_factor = np.random.uniform(0.5, 1.0, idv.shape)
    noise = np.random.normal(0, np.abs(diff_vector), idv.shape)
    
    mutation_spark = self.remap(idv + scale_factor * diff_vector + noise, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
