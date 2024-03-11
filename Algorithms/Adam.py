import numpy as np

from Algorithms.BaseOptimizer import BaseOptimizer


class Adam(BaseOptimizer):
    def __init__(self, num_features, step_size, beta_m, beta_v):
        self.m = np.zeros(num_features)
        self.v = np.zeros(num_features)

        self.step_size = step_size
        self.beta_m = beta_m
        self.beta_v = beta_v
        self.epsilon = 1e-8

        self.beta_m_product = beta_m
        self.beta_v_product = beta_v

    def compute_update(self, td_update):
        self.m = self.beta_m * self.m + (1 - self.beta_m) * td_update
        self.v = self.beta_v * self.v + (1 - self.beta_v) * (td_update * td_update)
        m_hat = self.m / (1 - self.beta_m_product)
        v_hat = self.v / (1 - self.beta_v_product)

        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        update_ = self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return update_
