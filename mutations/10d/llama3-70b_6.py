import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    r = np.random.rand()
    if r < 0.4:
        mutation_spark = idv + self.amps[0] * np.random.normal(0, 1, idv.shape[0])
    elif r < 0.8:
        idx = np.random.randint(0, num_spk)
        mutation_spark = idv + self.amps[0] * (spk_pop[idx] - idv)
    else:
        mutation_spark = idv + self.amps[0] * np.random.uniform(-1, 1, idv.shape[0])
    mutation_spark = self.remap(mutation_spark, self.lb, self.ub)
    return mutation_spark.reshape(1, -1)
