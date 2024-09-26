import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    mutate_prob = 0.1
    delta = np.zeros_like(idv)
    for i in range(self.dim):
        if np.random.rand() < mutate_prob:
            delta[i] = np.random.uniform(-self.ub, self.ub)
    mutation_spark = self.remap(idv + delta, self.lb, self.ub).reshape(1, -1)
    return mutation_spark
