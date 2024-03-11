import numpy as np

from Algorithms.BaseOptimizer import BaseOptimizer


class TD:
    def __init__(self, num_features, lmbda, optimizer: BaseOptimizer):
        self.optimizer = optimizer
        self.w = np.zeros(num_features)
        self.z = np.zeros(num_features)
        self.x_t = np.zeros(num_features)
        self.lmbda = lmbda

    def update(self, US, gamma, x_tp1):
        delta = US + gamma * np.dot(x_tp1, self.w) - np.dot(self.x_t, self.w)
        self.z = gamma * self.lmbda * self.z + self.x_t
        self.x_t = x_tp1
        self.w = self.w + self.optimizer(td_update=delta * self.z)

    def predict(self, x):
        return np.dot(x, self.w)

    def reinitialize_w(self, index):
        self.w[index] = 0.0
        self.z[index] = 0.0
