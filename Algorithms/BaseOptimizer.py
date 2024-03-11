from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    @abstractmethod
    def compute_update(self, **kwargs):
        raise NotImplementedError

    def __call__(self, **kwargs):
        return self.compute_update(**kwargs)
