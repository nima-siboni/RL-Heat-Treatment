from __future__ import annotations

import json
import tempfile
from os import listdir
from os import makedirs
from os import path
from shutil import rmtree
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from ray.tune.logger import UnifiedLogger
from ray.rllib.algorithms.dqn import dqn
from ray.rllib.utils.schedules import PiecewiseSchedule


def custom_log_creator(custom_path):
    def logger_creator(config):
        if not path.exists(custom_path):
            makedirs(custom_path)
        logdir = tempfile.mkdtemp(dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def create_or_clean_training_dirs(
    checkpoint_dir,
    logger_dir,
    clean_previous_checkpoints,
    clean_previous_logs,
):
    """
    creates or cleans the directories for the checkpoints and tensorboard logs

    :param checkpoint_dir: the checkpoint dir
    :param logger_dir: the directory for the tensorboard logs
    :param clean_previous_logs: a boolean, self-explanatory
    :param clean_previous_checkpoints: a boolean, self-explanatory
    """

    if clean_previous_checkpoints and path.exists(checkpoint_dir):
        print('the checkpoints directory is non-empty. It will be deleted.')
        rmtree(checkpoint_dir)

    if not path.exists(checkpoint_dir):
        print('The checkpoints directory does not exist. It will be created.')
        makedirs(checkpoint_dir)

    if clean_previous_logs and path.exists(logger_dir):
        print('the log directory is non-empty. It will be deleted.')
        rmtree(logger_dir)

    if not path.exists(logger_dir):
        print('The directory does not exist. It will be created.')
        makedirs(logger_dir)


def plot_the_pf(state, pause=0.2):
    """
    plots the phase field

    :param state: the phase field
    :param pause: the pause (in seconds) after the plotting
    """
    plt.clf()
    plt.title('T: {T:.2f}'.format(T=state['temperature']))
    plt.imshow(state['PF'][:, :, 0])
    plt.pause(pause)


def create_config(env_cfg_path: str, algorithm_cfg_path: str) -> Dict:
    """
    Read the env config and algorithm config and create the config dictionary needed for RLlib.

    Args:
         algorithm_cfg_path: path to json file which has the modification to the base dqn config.
         env_cfg_path: path to a json file which has the config of the env.
    Returns:
          the combined config.
    """
    config = dqn.DQNConfig().to_dict()
    env_config = json.load(open(env_cfg_path))
    config['env_config'] = env_config
    agent_config = json.load(open(algorithm_cfg_path))
    # agent_config['local_tf_session_args'] = {
    #     'intra_op_parallelism_threads': agent_config['num_workers'],
    #     'inter_op_parallelism_threads': agent_config['num_workers'],
    # }
    config.update(agent_config)

    # update the exploration config
    # config['exploration_config'].update(
    #     {  # Further Configs for the Exploration class' constructor:
    #         'epsilon_schedule': PiecewiseSchedule(
    #             endpoints=create_end_points(),
    #             framework='tf',
    #             outside_value=0.01,
    #         ),
    #     },
    # )
    return config


def load_training_config(training_cfg_path='./configs/'):
    """
    loads the config for training.
    :param training_cfg_path: the location of the config file
    training_config.json
    :returns: the training config as dictionary
    """
    training_config = json.load(open(training_cfg_path))
    if training_config['resume_training'] == 'False':
        training_config['resume_training'] = False
    if training_config['resume_training'] == 'True':
        training_config['resume_training'] = True

    if training_config['clean_previous_checkpoints'] == 'False':
        training_config['clean_previous_checkpoints'] = False
    if training_config['clean_previous_checkpoints'] == 'True':
        training_config['clean_previous_checkpoints'] = True

    if training_config['clean_previous_logs'] == 'False':
        training_config['clean_previous_logs'] = False
    if training_config['clean_previous_logs'] == 'True':
        training_config['clean_previous_logs'] = True

    if (
        training_config['resume_training']
        and training_config['clean_previous_checkpoints']
    ):
        assert not (
            training_config['restoring_dir'] ==
            training_config['checkpoint_dir']
        ), (
            'The code is going to clean the checkpoints first, '
            'then load one checkpoint to continue!'
        )

    return training_config


def find_the_checkpoint_with_largest_id(checkpoint_dir):
    """
    finds the checkpoint with the largest id
    :param checkpoint_dir: the directory of checkpoints
    :returns: the full path to the checkpoint which can be directly called by
     .restore from different Trainers in RLlib
    """
    all_dir_lst = listdir(checkpoint_dir)
    interesting_dir_lst = [
        dr for dr in all_dir_lst if (len(dr) > 10 and dr[:10] == 'checkpoint')
    ]
    dir_ids_as_int = np.array([
        dr[11:]
        for dr in interesting_dir_lst
    ]).astype(int)
    dir_id_with_largest_timestamps = np.argmax(dir_ids_as_int)
    largest_timestamp = np.max(dir_ids_as_int)
    checkpoint_dir = (
        checkpoint_dir
        + interesting_dir_lst[dir_id_with_largest_timestamps]
        + '/checkpoint-'
        + str(
            largest_timestamp,
        )
    )
    return checkpoint_dir


def create_end_points(
    nr_intervals: int = 20_000,
    interval_steps: int = 2_000,
    min_exploration: float = 0.01,
    decay_factor: float = 0.02,
) -> list:
    """
    Creating the end points for piecewise linear scheduling.

    The points are created to have an alternative high and low exploration rate
     the endpoints create many intervals where each has a high and low
     exploration phase. The value of the epsilon decays in the high exploration
     phase to its low exploration phase (which is a constant).

    :param nr_intervals: number of interval
    :param interval_steps: the time scale for each interval
    :param min_exploration: the min value for epsilon; make sure that this
     matches the outside value
    :param decay_factor: the factor which controls the decay of epsilon in high
     exploration phase decays following
     1/(decay factor * step + 1)

    :return: the endpoints for PiecewiseSchedule
    """
    endpoints = []
    for i in range(nr_intervals):
        endpoints.append(
            (
                interval_steps * (2 * i),
                np.max(
                    [
                        1.0 / (decay_factor * i + 1),
                        min_exploration,
                    ],
                ),
            ),
        )
        endpoints.append((interval_steps * (2 * i + 1), min_exploration))
    return endpoints
