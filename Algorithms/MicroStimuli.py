import numpy as np
from Algorithms.BaseRep import BaseRep


class MicroStimuli(BaseRep):
    def __init__(self, **kwargs):
        super(MicroStimuli, self).__init__(**kwargs)
        self.trace_list = np.zeros(self.total_signals)
        self.trace_parameter = kwargs["trace_parameter"]
        # === RBF parameters ===
        self.microstimuli_num = kwargs["num_trace_features"]
        self.rbf_std = .08
        self.mu = np.arange(1, self.microstimuli_num + 1) / self.microstimuli_num
        self.num_features = 1 + self.total_signals + self.total_signals * self.microstimuli_num
        self.stimuli_list_tm1 = np.zeros(self.total_signals)

    def get_feature_rep(self, stimuli_list, **kwargs):
        x = np.zeros(self.num_features)
        x[0] = 1
        # === Presence ===
        x[1: self.total_signals + 1] = stimuli_list
        # === stimuli onset replacing traces ===
        stimuli_onset = stimuli_list * (1 - self.stimuli_list_tm1)
        self.trace_list = self.trace_list * self.trace_parameter
        active_stimuli_onset = np.where(stimuli_onset == 1.0)
        self.trace_list[active_stimuli_onset] = 1.0

        # === RBF of the traces ===
        rbf_start = self.total_signals + 1
        for idx in range(self.total_signals):
            numerator = (self.mu - self.trace_list[idx]) * (self.trace_list[idx] - self.mu)
            denominator = 2 * self.rbf_std * self.rbf_std
            x[idx * self.microstimuli_num + rbf_start: (idx + 1) * self.microstimuli_num + rbf_start] = np.exp(
                numerator / denominator)
        self.stimuli_list_tm1 = stimuli_list
        return x
