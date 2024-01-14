from __future__ import annotations

import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
import ray
import ray.rllib.algorithms.dqn as dqn
from furnace import Furnace
from model.keras_model import KerasQModel
from ray.rllib.models import ModelCatalog

from utils.utils import create_config
from utils.utils import find_the_checkpoint_with_largest_id
from utils.utils import load_training_config

# ------------from keras_model import KerasQModel-------------------------
# 1 - Setting the configs
# 2 - Creating the agent in its env
# 3 - Testing the agent
# ---------------------------------------------
# ---------------------------------------------
# 1.1 -- The configuration of the optimization problem
# 1.1 -- The config of the DDQN trainer and its env.

nr_repetitions = 3
config = create_config(env_cfg_path="configs/env_config.json",
                       algorithm_cfg_path="./configs/agent_config.json")
training_config = load_training_config("./configs/training_config.json")
# import pdb; pdb.set_trace()
find_the_latest_agent = False

# the directory for outputting the performance analysis
results_directory = './agents_performance_analysis/'

# 1.1.0 -- register the keras model
ModelCatalog.register_custom_model('keras_Q_model', KerasQModel)

# 1.1.1 -- the DQN configs and the performance analysis results directory

if not path.exists(results_directory):
    os.makedirs(results_directory)


# ---------------------------------------------
# ---------------------------------------------
# 1.3 -- The checkpoint the trained agent and
# number of tests
# restored.

if find_the_latest_agent:
    # TODO: This part should be written better, as currently it finds the
    #  directory with the largest checkpoint which is not necessarily the
    #  newest one (for example if the check point directories are not cleaned
    #  at the beginning of the run
    checkpoint_dir = find_the_checkpoint_with_largest_id(
        training_config['checkpoint_dir'],
    )
    print('The following agent is recovered:\n', checkpoint_dir)
    # save the checkpoint directory to results_directory
    with open(results_directory + 'checkpoint_dir.txt', 'w') as f:
        f.write(checkpoint_dir)

else:
    checkpoint_dir = \
        'training_results/checkpoints/checkpoint_8200/'

assert path.exists(
    checkpoint_dir,
), 'the directory for checkpoints does not exist.'

# ---------------------------------------------
# Initialize  RAY
ray.init()
# ---------------------------------------------
# 2.1 -- Creating the trainer within its env.
config['explore'] = False
config['exploration'] = {'type': 'EpsilonGreedy'}
agent = dqn.DQNConfig().update_from_dict(config).environment(Furnace).build()

agent.restore(checkpoint_path=checkpoint_dir)
# policy = agent.get_policy()
# instantiate env class
env = Furnace(env_config=config['env_config'])
# ---------------------------------------------
# 3 - Testing
# run the test nr_repetitions and analyze the results.
mean_reward = 0
mean_squared_reward = 0

reward_lst = []  # rewards of only one experiemetn
obs_lst = []  # observation of only one experiement
action_lst = []  # actions of only one experiement
episode_reward_list = []
reward = 0
for exp_id in range(nr_repetitions):

    episode_reward = 0
    done = False
    obs_lst = []
    action_lst = []
    accumulated_reward_lst = []
    accumulated_reward = 0
    accumulated_energy_cost = 0
    accumulated_energy_cost_lst = []
    g2_lst = []
    # env.seed(seed=exp_id)
    obs, _ = env.reset()
    counter = 0
    obs_lst.append(obs)
    density_lst = []
    density = 0
    g2 = 0
    print('\nexp. id:', exp_id)
    while not done:
        action = agent.compute_single_action(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action=action)
        done = terminated or truncated
        accumulated_reward += reward
        action_lst.append(action)
        obs_lst.append(obs)
        accumulated_reward_lst.append(accumulated_reward)
        if not done:
            accumulated_energy_cost += info['energy_cost']
            g2 = info['g2']
            density = info['density']
        accumulated_energy_cost_lst.append(accumulated_energy_cost)
        g2_lst.append(g2)
        density_lst.append(density)
        counter += 1
    print('total number of steps:', counter, 'reward:', accumulated_reward)

    pf_lst = np.array([obs_lst[i]['PF'][:, :, 0] for i in range(len(obs_lst))])
    T_lst = np.array([
        obs_lst[i]['temperature'][0]
        for i in range(len(obs_lst))
    ])
    masses_lst = np.array(
        [
            np.mean(
                obs_lst[i]['PF'][:, :, 0] - config['env_config']['shift_pf'],
            )
            for i in range(len(obs_lst))
        ],
    )

    exp_dir = path.join(results_directory, str(exp_id))
    os.makedirs(exp_dir, exist_ok=True)

    # Plotting the final and initial pf
    plt.imshow(pf_lst[0])
    plt.title('Initial pf')
    plt.savefig(path.join(exp_dir, 'initial_pf.png'), format='png')
    plt.close()

    plt.imshow(pf_lst[-1])
    plt.title('Final pf')
    plt.savefig(path.join(exp_dir, 'final_pf.png'), format='png')
    plt.close()
    # plotting all the intermediate states as well
    pfs_dir = path.join(exp_dir, 'pfs')
    os.makedirs(pfs_dir, exist_ok=True)
    for step, pf in enumerate(pf_lst):
        if step < 10:
            filename = 'pf_' + '0000' + str(step) + '.png'
        elif step < 100:
            filename = 'pf_' + '000' + str(step) + '.png'
        elif step < 1000:
            filename = 'pf_' + '00' + str(step) + '.png'
        elif step < 10_000:
            filename = 'pf_' + '0' + str(step) + '.png'
        else:
            filename = 'pf_' + str(step) + '.png'
        if step % 10 == 0:
            plt.imshow(pf)
            plt.title('step: ' + str(step))
            # print("saved at", step)
            plt.savefig(path.join(pfs_dir, filename), format='png')
            plt.close()

    os.system(
        'convert -delay 30 -loop 0 '
        + str(pfs_dir)
        + '/pf_*.png '
        + exp_dir
        + '/pf_evolution.gif',
    )
    plt.plot(T_lst)
    plt.xlabel('steps')
    plt.ylabel('T')
    plt.savefig(path.join(exp_dir, 'temperature.png'), format='png')
    plt.close()
    # save the T as csv
    np.savetxt(
        path.join(exp_dir, 'temperature.csv'),
        np.array(T_lst),
        delimiter=',',
    )

    # plotting the densities
    plt.plot(masses_lst)
    plt.xlabel('steps')
    plt.ylabel('density')
    plt.savefig(path.join(exp_dir, 'density.png'), format='png')
    plt.close()
    np.savetxt(
        path.join(exp_dir, 'density.csv'),
        np.array(masses_lst),
        delimiter=',',
    )

    # plotting the g2
    plt.plot(g2_lst)
    plt.xlabel('steps')
    plt.ylabel('G2')
    # log y scale
    plt.yscale('log')
    plt.savefig(path.join(exp_dir, 'G2.png'), format='png')
    plt.close()
    np.savetxt(path.join(exp_dir, 'G2.csv'), np.array(g2_lst), delimiter=',')

    # plotting the energy cost
    plt.plot(accumulated_energy_cost_lst)
    plt.xlabel('steps')
    plt.ylabel('Energy cost')
    plt.savefig(path.join(exp_dir, 'energy_cost.png'), format='png')
    plt.close()
    np.savetxt(
        path.join(exp_dir, 'energy_cost.csv'),
        np.array(
            accumulated_energy_cost_lst,
        ),
        delimiter=',',
    )

    # plotting the rewards
    plt.plot(accumulated_reward_lst)
    plt.xlabel('steps')
    plt.ylabel('Acc. reward')
    plt.savefig(path.join(exp_dir, 'accumulated_reward.png'), format='png')
    plt.close()
    np.savetxt(
        path.join(exp_dir, 'accumulated_reward.csv'),
        np.array(accumulated_reward_lst),
        delimiter=',',
    )

ray.shutdown()
