import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    top_num = int(num_spk * self.gm_ratio)
    sort_idx = np.argsort(spk_fit)
    top_idx = sort_idx[:top_num]
    top_mean = np.mean(spk_pop[top_idx, :], axis=0)
    
    # Create a blend of the top solutions with the individual solution to enhance exploration
    alpha = np.random.uniform(0.2, 0.8, idv.shape)
    mutation_vector = alpha * idv + (1 - alpha) * top_mean

    # Introduce Gaussian noise to increase exploration
    std_dev = np.std(spk_pop, axis=0)
    noise = np.random.normal(0, std_dev, idv.shape)

    mutation_spark = self.remap(mutation_vector + noise, self.lb, self.ub).reshape(1, -1)
    
    return mutation_spark
