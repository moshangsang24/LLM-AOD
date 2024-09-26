import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    top_num = int(num_spk * self.gm_ratio)
    sort_idx = np.argsort(spk_fit)
    top_idx = sort_idx[:top_num]
    btm_idx = sort_idx[-top_num:]
    
    # Mean of top and bottom performers
    top_mean = np.mean(spk_pop[top_idx, :], axis=0)
    btm_mean = np.mean(spk_pop[btm_idx, :], axis=0)

    # Use a combination of top_mean, btm_mean and Gaussian noise
    alpha = np.random.uniform(0.4, 0.6, idv.shape)
    mutation_vector = alpha * top_mean + (1 - alpha) * btm_mean

    # Adding Gaussian noise for exploration
    std_dev = np.std(spk_pop, axis=0) + EPS  # Add EPS to avoid zero std_dev
    noise = np.random.normal(0, std_dev, idv.shape)

    mutation_spark = self.remap(mutation_vector + noise, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
