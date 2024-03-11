## Classical conditioning benchmarks for state construction
This repository provides the benchmarks inspired by classical conditioning for state construction presented in the [From eye-blinks to state construction: Diagnostic benchmarks for online representation learning](https://journals.sagepub.com/doi/full/10.1177/10597123221085039) paper and the [State Construction in Reinforcement Learning](https://drive.google.com/drive/search?q=thesis) PhD Thesis.


<a name='specifications'></a>
## Specification of Dependencies
The packages required for running the code are included in the `requirements.txt`
file. To install these dependencies, run the following command:
```text
pip install requirements.txt
```

## How to Use the Benchmarks
The code for the trace conditioning, noisy patterning, and trace patterning benchmarks are included in `classical_conditioning_benchmarks.py`. The benchmarks inherit from the `dm_env.Environment` class from the [The DeepMind RL Environment API](https://github.com/google-deepmind/dm_env) and implement its `reset()` and `step()` functions.

In order to use each of the benchmarks, you need to first instantiate an object from the class corresponding to the desired benchmark. Here is in example of instantiating an object `env` from the class corresponding to the trace conditioing benchmark:
```python
  env = TraceConditioning(seed=0,
                          gamma=0.9,
                          activation_lengths = {"CS": 4, "US":2, "distractor": 4},
                          ISI_interval=(7, 13),
                          ITI_interval=(80, 129),
                          num_distractors=0)
```

After instantiating `env`, you need to reset the environment by calling `env.reset()`, and then call `env.step()` function to get the information about the current time step such as the observation and cumulant:
```python
  env.reset()
  for t in range(config.num_time_steps-1):
      _, cumulant, _, obs = env.step(None)
```
Note that in the classial conditioning benchmarks, the cumulant is the unconditioned stimulus or US (See the paper for more information). Check out `exp.py` for a more detailed example of using the benchmarks.

`env.py` also includes a `compute_return_error()` function that given the series of unconditioned stimuli, $US_t$, and predictions, $\hat{v}_t$, computes the return, $G_t$, and the squared return error for every time step. The square return error at time step $t$ is defined as:

$$(\hat{v}_t - G_t)^2$$

## List of Publications
List of publications and submissions that used the classical conditioning benchmarks include:
- [Javed, K., Shah, H., Sutton, R. S., & White, M. (2023). Scalable real-time recurrent learning using columnar-constructive networks. Journal of Machine Learning Research](https://www.jmlr.org/papers/volume24/23-0367/23-0367.pdf)
- [Elelimy, E. M. (2023). Real Time Recurrent Learning with Complex-Valued Trace Units.](https://era.library.ualberta.ca/items/ecf6bafd-77cf-4e02-bd57-e2eadf46a6a6)
-  [Samani, A. (2022). Learning Agent State Online with Recurrent Generate-and-Test.](https://era.library.ualberta.ca/items/3603ecc0-79cd-486e-8b75-e703b20f7cae)

## Citation
If you found these benchmarks useful, please cite:

```
@article{rafiee2020eye,
  title={From Eye-blinks to State Construction: Diagnostic Benchmarks for Online Representation Learning},
  author={Rafiee, Banafsheh and Abbas, Zaheer and Ghiassian, Sina and Kumaraswamy, Raksha and Sutton, Richard and Ludvig, Elliot and White, Adam},
  journal={Adaptive Behavior},
  year={2022}
}
```
