import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    top_num = int(num_spk * self.gm_ratio)
    sort_idx = np.argsort(spk_fit)
    top_idx = sort_idx[:top_num]
    btm_idx = sort_idx[-top_num:]

    # Calculate top and bottom means
    top_mean = np.mean(spk_pop[top_idx, :], axis=0)
    btm_mean = np.mean(spk_pop[btm_idx, :], axis=0)
    delta = top_mean - btm_mean
    
    # Additional perturbation component with adaptive scaling
    perturbation = np.random.uniform(-1, 1, size=idv.shape)
    scale_factor = np.linalg.norm(delta) / (np.linalg.norm(perturbation) + EPS)
    new_spark = idv + delta + scale_factor * perturbation

    mutation_spark = self.remap(new_spark, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
