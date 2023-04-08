# Reinforcement Learning for Heat-treatment
This is a reinforcement learning solution for finding the optimal heat-treatment in material sciences. The reinforcement learning is done with RLlib on a 2D phase-field environment, [Furnace-Env](https://github.com/nima-siboni/Furnace-Env).
## 0 -- Installation

### 0.1 -- Creating the conda env
```buildoutcfg
conda create -n rl python=3.8
```

### 0.2 -- Installing the required packages
Activate the conda env:
```buildoutcfg
conda activate rl
```
### 0.3 -- Update your pip
```buildoutcfg
pip install --upgrade pip
```
### 0.4 -- Clone the project
```buildoutcfg
git clone git@github.com:nima-siboni/RL-Heat-Treatment.git
cd RL-Heat-Treatment
```

### 0.4 -- Install the requirements
Install the requirements in the env
```buildoutcfg
pip install -r requirements.txt
```

### 0.5 -- Modifying RLlib (for ```EpsilonGreedyCautious```)
If you want to use Caution-Epsilon-Greedy instead of the standard Epsilon-Greedy
of RLlib, you need to follow these 3 steps:
* First configure the exploration config in ```agent_config.cfg```:
```python
"exploration_config": {"type": "EpsilonGreedyCautious"} # or "EpsilonGreedy"
```
* Then place the ```epsilon_greedy_cautious.py``` where RLlib saves its explorations;
For example under conda environment directory followed by ```envs/rl/lib/python3.8/site-packages/ray/rllib/utils/exploration```
* In the same directory modify the ```__init__.py``` by importing the new epsilon scheduler and adding it to list of all explorations ```__all__```:
```python
from ray.rllib.utils.exploration.epsilon_greedy_cautious import EpsilonGreedyCautious

__all__ = [
    ...
    "EpsilonGreedyCautious",
    ...
]
```

### 0.6 -- Installing the package
It is an optional step. You can install this repository as a package by running
```python
pip install -e .
```
from the main directory of the repo.

Have fun!

## 1 -- Configurations
The configuration of the enviornment, the agent, and the training are done by the following files
* ```env_config.cfg```: a config file for the environment, for details visit [Furnace-Env](https://github.com/nima-siboni/Furnace-Env).
* ```agent_config.cfg```: here one can set the RLlib's algorithm config, and finally
* ```training_config.cfg```: common training/logging configs are set here.
## 2 -- Learning

To start learning, execute the following command within the ```RL-Heat-Treatment```:
```python experience and learn.py```

## 3 -- Evaluating the trained agent
To evaluate the agent's performance, in ```agents_performance.py```, set:
* the directory from where the agent should be loaded or you can alternatively set the
```find_the_latest_agent = True``` to load the last checkpoint:
```buildoutcfg
checkpoint_dir = 'training_results/checkpoints/checkpoint_000031/checkpoint-31'
```
* the directory the results should be outputted
```buildoutcfg
results_directory = './agents_performance_analysis/'
```
Executing this file, creates a directory with one subdirectory for each evaluation episode containing:
* the initial phase field,
* the final phase field,
* a GIF of intermediate PFs,
* temperature vs step profile,
* the density vs steps,
* energy cost vs steps,
* accumulated rewards vs steps, and
* G2 vs steps.

To have an statistical analysis of the performance of the agent over many episodes, use
```performance_analysis.py``` to produce histograms and joint plots of the performance of the trained agent.
