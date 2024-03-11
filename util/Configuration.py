import json
import os

default_params = {
    "ISI": 10,
    "starting_run": 0,
    "rep_name": "Presence",
    "step_size": 0.0005,
    "lmbda": 0.9,
    "num_trace_features": 16,
    "num_runs": 30,
    "num_time_steps": 2000000,
    "beta_m": 0.9,
    "beta_v": 0.999,
    "num_poisson_distractor": 0,
    "num_distractor_coinciding_CS": 0,
    "num_CS": 2,
    "num_activation_patterns": 1,
    "noise": 0.1
}


def find_all_experiment_configuration(experiments_path: str):
    if experiments_path.endswith('.json'):
        yield experiments_path
    for root, _, files in os.walk(experiments_path):
        for file in files:
            if file.endswith('.json'):
                yield os.path.join(root, file)


class Configuration:
    def __init__(self, json_path: str = None, args: dict = None):
        if json_path is not None:
            self._path = json_path
            with open(self._path) as f:
                self._params = json.load(f)
        elif args is not None:
            self._params = args

    @property
    def testbed_name(self):
        return self._params['testbed_name']

    @property
    def ISI(self):
        return self._params.get('ISI', default_params['ISI'])

    @property
    def starting_run(self):
        return self._params.get('starting_run', default_params['starting_run'])

    @property
    def rep_name(self):
        return self._params.get('rep_name', default_params['rep_name'])

    @property
    def step_size(self):
        return self._params.get('step_size', default_params['step_size'])

    @property
    def lmbda(self):
        return self._params.get('lmbda', default_params['lmbda'])

    @property
    def num_trace_features(self):
        return self._params.get('num_trace_features', default_params['num_trace_features'])

    @property
    def num_runs(self):
        return self._params.get('num_runs', default_params['num_runs'])

    @property
    def num_time_steps(self):
        return self._params.get('num_time_steps', default_params['num_time_steps'])

    @property
    def beta_m(self):
        return self._params.get('beta_m', default_params['beta_m'])

    @property
    def beta_v(self):
        return self._params.get('beta_v', default_params['beta_v'])

    @property
    def num_poisson_distractor(self):
        return self._params.get('num_poisson_distractor', default_params['num_poisson_distractor'])

    @property
    def num_distractor_coinciding_CS(self):
        return self._params.get('num_distractor_coinciding_CS', default_params['num_distractor_coinciding_CS'])

    @property
    def num_CS(self):
        return self._params.get('num_CS', default_params['num_CS'])

    @property
    def num_activation_patterns(self):
        return self._params.get('num_activation_patterns', default_params['num_activation_patterns'])

    @property
    def noise(self):
        return self._params.get('noise', default_params['noise'])
