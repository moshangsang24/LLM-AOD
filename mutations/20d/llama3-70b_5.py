import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    mutate_prob = 0.3
    delta = np.zeros_like(idv)
    if np.random.rand() < mutate_prob:
        idx = np.random.randint(0, num_spk)
        delta = idv - spk_pop[idx]
        delta *= np.random.uniform(0.9, 1.1)
    else:
        for i in range(self.dim):
            if np.random.rand() < 0.15:
                delta[i] = np.random.uniform(-self.ub, self.ub)
            elif np.random.rand() < 0.05:
                delta[i] = np.random.normal(0, 0.1)
    mutation_spark = self.remap(idv + delta, self.lb, self.ub).reshape(1, -1)
    return mutation_spark
