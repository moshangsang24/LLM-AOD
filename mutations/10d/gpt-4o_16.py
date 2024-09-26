import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    sorted_idx = np.argsort(spk_fit)
    
    # Select the top performers
    top_idx = sorted_idx[:int(num_spk * self.gm_ratio)]
    top_centroid = np.mean(spk_pop[top_idx, :], axis=0)
    
    # Compute differential vectors
    diff_vectors = spk_pop[top_idx] - idv
    diff_vector = np.mean(diff_vectors, axis=0)
    
    # Add scaled differential vector, Gaussian noise, and hybrid perturbation
    scale_factor = np.random.uniform(0.4, 0.9, idv.shape)
    noise = np.random.normal(0, np.abs(diff_vector), idv.shape)
    hybrid_perturbation = np.random.uniform(-0.1, 0.1, idv.shape) * np.random.randn(*idv.shape)
    
    mutation_spark = self.remap(idv + scale_factor * diff_vector + noise + hybrid_perturbation, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
