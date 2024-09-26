import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    sorted_idx = np.argsort(spk_fit)
    
    # Select the top and bottom performers
    top_idx = sorted_idx[:int(num_spk * self.gm_ratio)]
    btm_idx = sorted_idx[-int(num_spk * self.gm_ratio):]
    
    # Compute centroids for top and bottom performers
    top_centroid = np.mean(spk_pop[top_idx, :], axis=0)
    btm_centroid = np.mean(spk_pop[btm_idx, :], axis=0)
    
    # Differential vector between top and bottom centroids
    diff_vector = top_centroid - btm_centroid

    # Introduce scaling factor, Gaussian noise, and random walk
    scale_factor = np.random.uniform(0.3, 0.7, idv.shape)
    noise = np.random.normal(0, np.abs(diff_vector), idv.shape)
    walk_step = np.random.uniform(-1, 1, idv.shape)
    
    mutation_spark = self.remap(idv + scale_factor * diff_vector + noise + walk_step, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
