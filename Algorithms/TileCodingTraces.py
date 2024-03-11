import numpy as np
import Algorithms.tiles3 as tc
from Algorithms.BaseRep import BaseRep


class TileCodingTraces(BaseRep):
    def __init__(self, **kwargs):
        super(TileCodingTraces, self).__init__(**kwargs)
        self.trace_list = np.zeros(self.total_signals)
        self.trace_parameter = kwargs["trace_parameter"]
        # === Tile Coding Parameters ===
        self.num_tilings = 2
        # self.num_tilings = 1        # plot TCT rep
        self.iht_size = kwargs["num_trace_features"]
        self.tile_scale = (self.iht_size / self.num_tilings)  # trace scale is 1.0
        self.iht_list = [tc.IHT(self.iht_size) for _ in range(self.total_signals)]
        self.stimuli_list_tm1 = np.zeros(self.total_signals)
        self.num_features = 1 + self.total_signals + self.total_signals * self.iht_size
        # self.num_features = self.iht_size # plot TCT rep

    def get_feature_rep(self, stimuli_list, **kwargs):
        x = np.zeros(self.num_features)
        x[0] = 1
        x[1: self.total_signals + 1] = stimuli_list

        # === stimuli onset replacing traces ===
        stimuli_onset = stimuli_list * (1 - self.stimuli_list_tm1)
        self.trace_list = self.trace_list * self.trace_parameter
        self.trace_list[stimuli_onset == 1.0] = 1.0

        for idx in range(self.total_signals):
            active_tiles = np.array(tc.tiles(self.iht_list[idx], self.num_tilings,
                                    np.array([self.trace_list[idx] * self.tile_scale])))
            x[idx * self.iht_size + self.total_signals + 1 + active_tiles] = 1

        # # plot TCT rep start
        # active_tiles = np.array(tc.tiles(self.iht_list[0], self.num_tilings,
        #                         np.array([self.trace_list[1] * self.tile_scale])))
        # x[active_tiles] = 1
        # # end

        self.stimuli_list_tm1 = stimuli_list
        return x
