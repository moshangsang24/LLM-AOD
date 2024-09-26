import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    sorted_idx = np.argsort(spk_fit)
    
    # Select the best performers
    best_idx = sorted_idx[:int(num_spk * self.gm_ratio)]
    best_centroid = np.mean(spk_pop[best_idx, :], axis=0)
    
    # Randomly select other individuals for diversity
    rand_idx = np.random.choice(num_spk, int(num_spk * self.gm_ratio), replace=False)
    rand_centroid = np.mean(spk_pop[rand_idx, :], axis=0)
    
    # Compute differential vector
    diff_vector = best_centroid - rand_centroid
    
    # Add scaled differential vector and Gaussian noise
    scale_factor = np.random.uniform(0.5, 1.0, idv.shape)
    noise = np.random.normal(0, np.abs(diff_vector), idv.shape)
    
    mutation_spark = self.remap(idv + scale_factor * diff_vector + noise, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
