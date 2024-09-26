import numpy as np
import os

def mutate_operation(self, idv, spk_pop, spk_fit):
    num_spk = spk_pop.shape[0]
    top_num = int(num_spk * self.gm_ratio)
    sort_idx = np.argsort(spk_fit)
    top_idx = sort_idx[:top_num]
    btm_idx = sort_idx[-top_num:]

    # Top and bottom mean individuals
    top_mean = np.mean(spk_pop[top_idx, :], axis=0)
    btm_mean = np.mean(spk_pop[btm_idx, :], axis=0)
    delta = top_mean - btm_mean

    # Combination of Gaussian, Uniform, Levy flight, Differential Evolution (DE) perturbations, Simulated Annealing (SA), Swarm Intelligence, and Adaptive Component
    gaussian_perturbation = np.random.normal(0, 1, size=idv.shape)
    uniform_perturbation = np.random.uniform(-1, 1, size=idv.shape)

    beta = 1.5
    sigma = (
        np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    levy_perturbation = np.random.normal(
        0, sigma, size=idv.shape
    ) / (np.abs(np.random.normal(0, 1, size=idv.shape))) ** (1 / beta)

    # Differential Evolution (DE) component
    f = 0.5  # DE scaling factor
    r1, r2 = np.random.randint(0, num_spk, size=2)
    while r1 == r2:
        r2 = np.random.randint(0, num_spk)
    de_perturbation = f * (spk_pop[r1, :] - spk_pop[r2, :])

    # Simulated Annealing component
    T = 1.0  # initial temperature
    alpha = 0.9  # cooling rate
    sa_perturbation = np.random.uniform(-T, T, size=idv.shape)
    T *= alpha  # cooling

    # Swarm Intelligence component (influence of best individual in the neighborhood)
    best_individual_idx = np.argmin(spk_fit)
    best_individual = spk_pop[best_individual_idx, :]
    swarm_influence = best_individual - idv

    # Adaptive component to balance exploration and exploitation
    performance_ratio = (self.cur_gen / self.max_gen) ** 2  # Adjusting factor based on progression
    exploration_component = (
        0.25 * gaussian_perturbation
        + 0.25 * uniform_perturbation
        + 0.25 * levy_perturbation
    )
    exploitation_component = (
        0.1 * de_perturbation + 0.1 * sa_perturbation + 0.05 * swarm_influence
    )

    combined_perturbation = (
        (1 - performance_ratio) * exploration_component
        + performance_ratio * exploitation_component
    )
    new_spark = idv + delta + combined_perturbation

    mutation_spark = self.remap(new_spark, self.lb, self.ub).reshape(1, -1)

    return mutation_spark
