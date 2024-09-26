import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    top_num = int(num_spk * self.gm_ratio)
    
    sort_idx = np.argsort(spk_fit)
    top_idx = sort_idx[:top_num]
    btm_idx = sort_idx[-top_num:]
    
    top_mean = np.mean(spk_pop[top_idx, :], axis=0)
    btm_mean = np.mean(spk_pop[btm_idx, :], axis=0)
    
    diff = top_mean - btm_mean
    scale = np.random.uniform(low=0.6, high=1.7, size=len(diff))
    
    mutation_spark = idv + scale * diff
    mutation_spark = self.remap(mutation_spark, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
