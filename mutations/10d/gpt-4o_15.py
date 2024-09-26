import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    sorted_idx = np.argsort(spk_fit)
    
    # Select the top performers
    top_idx = sorted_idx[:int(num_spk * self.gm_ratio)]
    top_centroid = np.mean(spk_pop[top_idx, :], axis=0)
    
    # Select the bottom performers
    bottom_idx = sorted_idx[-int(num_spk * self.gm_ratio):]
    bottom_centroid = np.mean(spk_pop[bottom_idx, :], axis=0)
    
    # Differential vector between top and bottom centroids
    diff_vector = top_centroid - bottom_centroid
    
    # Gaussian perturbation based on both differential vector and initial idv
    scale_factor = np.random.uniform(0.4, 0.9, idv.shape)
    noise = np.random.normal(0, np.abs(diff_vector), idv.shape)
    perturbation = np.random.normal(0, 0.1 * np.abs(idv - diff_vector), idv.shape)
    
    mutation_spark = self.remap(idv + scale_factor * diff_vector + noise + perturbation, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
