import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    sorted_idx = np.argsort(spk_fit)
    
    # Select the best performers
    best_idx = sorted_idx[:int(num_spk * self.gm_ratio)]
    best_centroid = np.mean(spk_pop[best_idx, :], axis=0)
    
    # Select the worst performers
    worst_idx = sorted_idx[-int(num_spk * self.gm_ratio):]
    worst_centroid = np.mean(spk_pop[worst_idx, :], axis=0)
    
    # Compute differential vectors
    diff_vector_best = best_centroid - idv
    diff_vector_worst = idv - worst_centroid
    
    # Combine differential vectors
    combined_diff_vector = diff_vector_best + diff_vector_worst
    
    # Add scaled combined differential vector, Gaussian noise, and random perturbation
    scale_factor = np.random.uniform(0.4, 0.8, idv.shape)
    noise = np.random.normal(0, np.abs(combined_diff_vector), idv.shape)
    perturbation = np.random.uniform(-0.1 * np.abs(combined_diff_vector), 0.1 * np.abs(combined_diff_vector), idv.shape)
    
    mutation_spark = self.remap(idv + scale_factor * combined_diff_vector + noise + perturbation, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
