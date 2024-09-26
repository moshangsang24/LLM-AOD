import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
        num_spk = spk_pop.shape[0]
        top_num = int(num_spk * self.gm_ratio)
        sort_idx = np.argsort(spk_fit)
        top_idx = sort_idx[:top_num]
        top_spks = spk_pop[top_idx]
        num_dim = idv.shape[0]

        mutation_spark = np.zeros(num_dim)
        for i in range(num_dim):
            top_mean = np.mean(top_spks[:, i])
            mutation_spark[i] = idv[i] + 0.2 * (idv[i] - top_mean)

        mutation_spark = self.remap(mutation_spark, self.lb, self.ub).reshape(1, -1)
        
        return mutation_spark
