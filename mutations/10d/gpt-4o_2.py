import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    top_num = int(num_spk * self.gm_ratio)
    sort_idx = np.argsort(spk_fit)
    top_idx = sort_idx[:top_num]
    btm_idx = sort_idx[-top_num:]
    
    # Find the mean of the top and bottom performers
    top_mean = np.mean(spk_pop[top_idx, :], axis=0)
    btm_mean = np.mean(spk_pop[btm_idx, :], axis=0)
    
    # Mutation strategy: weighted mean of top and bottom, plus Gaussian noise
    weight = 0.5
    combined_mean = weight * top_mean + (1 - weight) * btm_mean
    std_dev = np.std(spk_pop, axis=0)
    noise = np.random.normal(0, std_dev, idv.shape)
    
    mutation_spark = self.remap(idv + combined_mean + noise, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
