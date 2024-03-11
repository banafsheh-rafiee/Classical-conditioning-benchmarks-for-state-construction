import numpy as np
import argparse
import os

from Registry.AlgRegistry import *
from util.Configuration import Configuration
from classical_conditioning_benchmarks import *

if __name__ == '__main__':
    import time
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-p', type=str, default=None)
    parser.add_argument('--testbed_name', '-tn', type=str, default='trace_conditioning')
    parser.add_argument('--ISI', '-i', type=int, default=10)
    parser.add_argument('--starting_run', '-r', type=int, default=0)
    parser.add_argument('--rep_name', '-rn', type=str, default='Presence')
    parser.add_argument('--step_size', '-a', type=float, default=0.001)
    parser.add_argument('--lmbda', '-l', type=float, default=0.9)
    parser.add_argument('--num_trace_features', '-ntf', type=int, default=16)
    parser.add_argument('--num_runs', '-nr', type=int, default=10)
    parser.add_argument('--num_time_steps', '-nt', type=int, default=2000)
    parser.add_argument('--beta_m', '-bm', type=float, default=0.9)
    parser.add_argument('--beta_v', '-bv', type=float, default=0.999)
    parser.add_argument('--num_poisson_distractor', '-pd', type=int, default=0)
    parser.add_argument('--num_distractor_coinciding_CS', '-dcc', type=int, default=0)
    parser.add_argument('--num_CS', '-csn', type=int, default=1)
    parser.add_argument('--num_activation_patterns', '-ap', type=int, default=1)
    parser.add_argument('--noise', '-n', type=float, default=0.0)
    args = parser.parse_args()

    config = Configuration(json_path=args.config_path, args=vars(args))
    
    rep_dict = {"TileCodingTraces": TileCodingTraces,  
                "Presence": Presence, 
                "MicroStimuli": MicroStimuli}  
    trace_dict = {4: 0.75, 5: 0.8, 10: 0.9, 20: 0.95, 30: 0.966, 40: 0.975, 80: 0.9875}
    gamma_dict = {4: 0.75, 5: 0.8, 10: 0.9, 20: 0.95, 30: 0.966, 40: 0.975, 80: 0.9875}
    total_signals = 1 + config.num_CS + config.num_poisson_distractor + config.num_distractor_coinciding_CS

    rep_param = {'total_signals': total_signals,
                 'trace_parameter': trace_dict[config.ISI]**2,
                 'num_trace_features': config.num_trace_features}

    rep_unit = rep_dict[config.rep_name](**rep_param)
    inter_trial_interval = (80, 120)

    for run in range(config.starting_run, config.starting_run + config.num_runs):

        if config.testbed_name == "trace_conditioning":
            env = TraceConditioning(seed=run,
                                    gamma=gamma_dict[config.ISI],
                                    activation_lengths = {"CS": 4, "US":2, "distractor": 4},
                                    ISI_interval=(
                                    config.ISI - np.floor(config.ISI / 3), config.ISI + np.floor(config.ISI / 3)),
                                    ITI_interval=inter_trial_interval,
                                    num_distractors=config.num_poisson_distractor)
        elif config.testbed_name == "noisy_patterning":
            env = NoisyPatterning(seed=run,
                                  gamma=gamma_dict[config.ISI],
                                  activation_lengths = {"CS": 4, "US":2, "distractor": 4},
                                  ISI_interval=(config.ISI, config.ISI),
                                  ITI_interval=inter_trial_interval,
                                  num_distractors=config.num_distractor_coinciding_CS,
                                  num_CS=config.num_CS,
                                  num_activation_patterns=config.num_activation_patterns,
                                  activation_patterns_prob=0.5,
                                  noise=config.noise)
        elif config.testbed_name == "trace_patterning":
            env = TracePatterning(seed=run,
                                  gamma=gamma_dict[config.ISI],
                                  activation_lengths = {"CS": 4, "US":2, "distractor": 4},
                                  ISI_interval=(
                                      config.ISI - np.floor(config.ISI / 3), config.ISI + np.floor(config.ISI / 3)),
                                  ITI_interval=inter_trial_interval,
                                  num_distractors=config.num_distractor_coinciding_CS,
                                  num_CS=config.num_CS,
                                  num_activation_patterns=config.num_activation_patterns,
                                  activation_patterns_prob=0.5,
                                  noise=config.noise)
        env.reset()

        optimizer = Adam(num_features=rep_unit.num_features, step_size=config.step_size, beta_m=config.beta_m,
                         beta_v=config.beta_v)
        learner = TD(num_features=rep_unit.num_features, lmbda=config.lmbda, optimizer=optimizer)
        trial = 0

        signals_over_time = np.zeros((total_signals, config.num_time_steps))
        signals_over_time[:, 0] = env.observation()
        prediction_over_time = np.zeros(config.num_time_steps)
        error_over_time = np.zeros(config.num_time_steps)
        rewards = np.zeros(config.num_time_steps)
        CS_onsets = []

        for t in range(config.num_time_steps-1):
            _, reward, _, obs = env.step(None)

            # === constructing feature vector ===
            X = rep_unit.get_feature_rep(obs, **{"pred_tm1": prediction_over_time[t]})

            # === update logged data ===
            prediction = learner.predict(X)
            prediction_over_time[t+1] = prediction
            rewards[t] = reward
            signals_over_time[:, t + 1] = obs
            if sum(obs[1:config.num_CS+1]) >= 1 and sum(signals_over_time[1:config.num_CS+1, t]) == 0:
                CS_onsets.append(t)
            
            # === learning ===
            learner.update(reward, gamma_dict[config.ISI], X)

        
        MSRE, error_over_time, return_over_time = compute_return_error(rewards, prediction_over_time, gamma_dict[config.ISI],
                                                                 )
        if not os.path.exists("results/{}".format(config.testbed_name)):
            os.makedirs("results/{}".format(config.testbed_name))

        
        np.savez(
                "results/{}/{}".format(config.testbed_name, run),
                error_over_time = error_over_time,
                MSRE = np.mean(error_over_time),
                prediction_over_time = prediction_over_time,
                signals_over_time = signals_over_time,
                CS_onsets = CS_onsets,
                return_over_time = return_over_time
            )

