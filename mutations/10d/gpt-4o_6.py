import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    top_num = int(num_spk * self.gm_ratio)
    sort_idx = np.argsort(spk_fit)
    top_idx = sort_idx[:top_num]
    btm_idx = sort_idx[-top_num:]
    
    # Use both top and bottom performers to enhance exploration
    top_mean = np.mean(spk_pop[top_idx, :], axis=0)
    btm_mean = np.mean(spk_pop[btm_idx, :], axis=0)
    
    # Calculate blend of top and bottom means
    alpha = np.random.uniform(0.4, 0.6, idv.shape)
    mutation_vector = alpha * top_mean + (1 - alpha) * btm_mean

    # Introduce Gaussian noise and a small perturbation to increase exploration
    epsilon = 1e-8
    std_dev = np.std(spk_pop, axis=0) + epsilon
    noise = np.random.normal(0, std_dev, idv.shape)
    perturbation = np.random.uniform(-std_dev, std_dev, idv.shape)

    mutation_spark = self.remap(mutation_vector + noise + perturbation, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
