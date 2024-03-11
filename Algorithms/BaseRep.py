from abc import abstractmethod


class BaseRep:
    def __init__(self, **kwargs):
        self.total_signals = kwargs["total_signals"]

    @abstractmethod
    def get_feature_rep(self, stimuli_list, **kwargs):
        raise NotImplementedError
