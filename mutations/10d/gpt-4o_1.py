import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    top_num = int(num_spk * self.gm_ratio)
    sort_idx = np.argsort(spk_fit)
    top_idx = sort_idx[:top_num]
    top_mean = np.mean(spk_pop[top_idx, :], axis=0)
    
    # Introduce Gaussian noise and incorporate a random walk strategy to increase exploration
    std_dev = np.std(spk_pop[top_idx, :], axis=0)
    noise = np.random.normal(0, std_dev, idv.shape)
    
    # Random walk strategy
    walk_step = np.random.uniform(-std_dev, std_dev, idv.shape)
    
    mutation_spark = self.remap(idv + top_mean + noise + walk_step, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
