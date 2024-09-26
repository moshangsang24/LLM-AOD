import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    mutate_prob = 0.2
    delta = np.zeros_like(idv)
    if np.random.rand() < mutate_prob:
        p = np.random.uniform(0, 1)
        if p < 0.5:
            idx = np.random.randint(0, num_spk)
            delta = idv - spk_pop[idx]
            delta *= np.random.uniform(0.5, 1.5)
        else:
            delta = np.random.normal(0, 0.1, self.dim)
    mutation_spark = self.remap(idv + delta, self.lb, self.ub).reshape(1, -1)
    return mutation_spark
