from typing import List

import matplotlib.pyplot as plt
import json
import numpy as np
from ray.tune.logger import Logger, UnifiedLogger
import tempfile
from os import path, makedirs, listdir
from shutil import rmtree


def custom_log_creator(custom_path):
    def logger_creator(config):
        if not path.exists(custom_path):
            makedirs(custom_path)
        logdir = tempfile.mkdtemp(dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def create_or_clean_training_dirs(checkpoint_dir, logger_dir, clean_previous_checkpoints, clean_previous_logs):
    """
    creates or cleans the dorectories for the checkpoints and tensorboard logs
    
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


def load_env_config(env_config_dir='./'):
    """
    loads the environment config from the env_config.cfg

    :param env_config_dir: the dir in which the env_config.cfg is.
    :return:
    env_config as a dictionary.
    """
    env_config = json.load(open(env_config_dir + 'env_config.cfg'))

    if env_config["termination_change_criterion"] == "None":
        env_config["termination_change_criterion"] = None

    if env_config["termination_temperature_criterion"] == "True":
        env_config["termination_temperature_criterion"] = True
    if env_config["termination_temperature_criterion"] == "False":
        env_config["termination_temperature_criterion"] = False

    if env_config["verbose"] == "True":
        env_config["verbose"] = True

    if env_config["verbose"] == "False":
        env_config["verbose"] = False

    if env_config["stop_action"] == "False":
        env_config["stop_action"] = False

    if env_config["stop_action"] == "True":
        env_config["stop_action"] = True

    env_config["G_list"] = np.fromstring(env_config["G_list"], dtype=float, sep=',')
    assert len(env_config["G_list"]) > 0, 'Are you sure about G_list? there might be extra []'
    return env_config


def load_agent_config(agent_cfg_dir='./'):
    """
    reads the dictionary in agent_cfg.cfg and returns the dictionary.

    :param agent_cfg_dir: the address of the directory which has agent_cfg.cfg
    :return:
    the dictionary for agent's config
    """
    agent_config = json.load(open(agent_cfg_dir + 'agent_config.cfg'))
    return agent_config


def create_config(base_config, env_cfg_dir='./', agent_cfg_dir='./'):
    """
    Creates the config dictionary needed for the RLlib
    
    :param base_config: the config to start with (for example the DDQN or PPO default config)
    :param env_cfg_dir: the directory which has env_config.cfg
    :param agent_cfg_dir: the directory which has agent_config.cfg
    """
    env_config = load_env_config(env_config_dir=env_cfg_dir)
    agent_config = load_agent_config(agent_cfg_dir=agent_cfg_dir)
    agent_config["local_tf_session_args"] = {
        "intra_op_parallelism_threads": agent_config["num_workers"],
        "inter_op_parallelism_threads": agent_config["num_workers"]}
    base_config.update(agent_config)
    base_config['env_config'] = env_config
    return base_config


def load_training_config(training_cfg_dir='./'):
    """
    loads the config for training.
    :param training_cfg_dir: the location of the config file training_config.cfg
    :returns: the training config as dictionary
    """
    training_config = json.load(open(training_cfg_dir + 'training_config.cfg'))
    if training_config["resume_training"] == "False":
        training_config["resume_training"] = False
    if training_config["resume_training"] == "True":
        training_config["resume_training"] = True

    if training_config["clean_previous_checkpoints"] == "False":
        training_config["clean_previous_checkpoints"] = False
    if training_config["clean_previous_checkpoints"] == "True":
        training_config["clean_previous_checkpoints"] = True

    if training_config["clean_previous_logs"] == "False":
        training_config["clean_previous_logs"] = False
    if training_config["clean_previous_logs"] == "True":
        training_config["clean_previous_logs"] = True

    if training_config["resume_training"] and training_config["clean_previous_checkpoints"]:
        assert not (training_config["restoring_dir"] == training_config[
            "checkpoint_dir"]), "The code is going to clean the checkpoints first, then load one checkpoint to continue!"

    return training_config


def find_the_checkpoint_with_largest_id(checkpoint_dir):
    """
    finds the checkpoint with the largest id
    :param checkpoint_dir: the directory of checkpoints
    :returns: the full path to the checkpoint which can be directly called by .restore from different Trainers in RLlib
    """
    all_dir_lst = listdir(checkpoint_dir)
    interesting_dir_lst = [dr for dr in all_dir_lst if (len(dr) > 10 and dr[:10] == 'checkpoint')]
    dir_ids_as_int = np.array([dr[11:] for dr in interesting_dir_lst]).astype(int)
    dir_id_with_largest_timestamps = np.argmax(dir_ids_as_int)
    largest_timestamp = np.max(dir_ids_as_int)
    checkpoint_dir = checkpoint_dir + interesting_dir_lst[dir_id_with_largest_timestamps] + '/checkpoint-' + str(
        largest_timestamp)
    return checkpoint_dir


def create_end_points(nr_intervals: int = 20_000, interval_steps: int = 2_000, min_exploration: float = 0.01,
                      decay_factor: float = 0.02) -> List:
    """
    Creating the end points for piecewise linear scheduling.

    The points are created to have an alternative high and low exploration rate; the endpoints create many intervals
    where each has a high and low exploration phase. The value of the epsilon decays in the high exploration phase to
    its low exploration phase (which is a constant).

    :param nr_intervals: number of interval
    :param interval_steps: the time scale for each interval
    :param min_exploration: the min value for epsilon; make sure that this matches the outside value
    :param decay_factor: the factor which controls the decay of epsilon in high exploration phase decays following
     1/(decay factor * step + 1)

    :return: the endpoints for PiecewiseSchedule
    """
    endpoints = []
    for i in range(nr_intervals):
        endpoints.append(
            (
                interval_steps * (2 * i), np.max([1.0 / (decay_factor * i + 1), min_exploration])
            )
        )
        endpoints.append((interval_steps * (2 * i + 1), min_exploration))
    return endpoints
