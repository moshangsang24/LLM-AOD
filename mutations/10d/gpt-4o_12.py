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
    
    # Compute differential vector
    diff_vector = top_centroid - bottom_centroid
    
    # Introduce scaling factor and Gaussian noise
    scale_factor = np.random.uniform(0.3, 0.8, idv.shape)
    noise = np.random.normal(0, np.abs(diff_vector), idv.shape)
    
    # Adaptive perturbation
    rand_scale = np.random.uniform(0.1, 0.5, idv.shape)
    perturbation = rand_scale * np.random.normal(0, np.abs(diff_vector), idv.shape)
    
    mutation_spark = self.remap(idv + scale_factor * diff_vector + noise + perturbation, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
