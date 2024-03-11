import numpy as np
from Algorithms.BaseRep import BaseRep


class Presence(BaseRep):
    def __init__(self, **kwargs):
        super(Presence, self).__init__(**kwargs)
        self.num_features = 1 + self.total_signals

    def get_feature_rep(self, stimuli_list, **kwargs):
        x = np.zeros(self.num_features)
        x[0] = 1
        x[1:] = stimuli_list
        return x
