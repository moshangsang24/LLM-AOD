import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    top_num = int(num_spk * self.gm_ratio)
    sort_idx = np.argsort(spk_fit)
    top_idx = sort_idx[:top_num]

    # Select random individuals as well
    rand_idx = np.random.choice(range(num_spk), size=top_num, replace=False)
    
    top_mean = np.mean(spk_pop[top_idx, :], axis=0)
    rand_mean = np.mean(spk_pop[rand_idx, :], axis=0)
    
    delta = top_mean - rand_mean
    mutation_spark = self.remap(idv + delta, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
