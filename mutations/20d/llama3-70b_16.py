import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    mutate_prob = 0.22
    delta = np.zeros_like(idv)
    if np.random.rand() < mutate_prob:
        idx1, idx2 = np.random.randint(0, num_spk, 2)
        delta = spk_pop[idx1] - spk_pop[idx2]
        delta *= np.random.uniform(0.85, 1.15)
    else:
        for i in range(self.dim):
            if np.random.rand() < 0.09:
                delta[i] = np.random.uniform(-self.ub, self.ub)
            elif np.random.rand() < 0.03:
                delta[i] = np.random.normal(0, 0.08)
    mutation_spark = self.remap(idv + delta, self.lb, self.ub).reshape(1, -1)
    return mutation_spark
