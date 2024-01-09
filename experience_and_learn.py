from __future__ import annotations

import ray
import ray.rllib.algorithms.dqn as dqn
# from Furnace import Furnace
from ray.rllib.models import ModelCatalog

from ray.tune.logger import pretty_print

from furnace import Furnace
from model.keras_model import KerasQModel

from utils.utils import create_config, custom_log_creator

from utils.utils import create_or_clean_training_dirs

from utils.utils import load_training_config

# ---------------------------------------------
# 1 - Setting the configs
# 2 - Creating the agent in its env
# 3 - training and saving
# ---------------------------------------------

# 1.1 -- The config of the DDQN trainer and its env.

# 1.1.0 -- register the keras model
keras_q_model = KerasQModel
ModelCatalog.register_custom_model('keras_Q_model', keras_q_model)

# 1.1.1 -- the DQN configs

config = create_config(env_cfg_path="configs/env_config.json",
                       algorithm_cfg_path="./configs/agent_config.json")

# ---------------------------------------------


# 1.2 -- Rounds of training and Check point parameters
# (if and how often to write the network)
training_config = load_training_config("./configs/training_config.json")

nr_training_rounds = training_config['nr_training_rounds']

checkpoint_interval = training_config['checkpoint_interval']

create_or_clean_training_dirs(
    checkpoint_dir=training_config['checkpoint_dir'],
    logger_dir=training_config['logger_dir'],
    clean_previous_checkpoints=training_config['clean_previous_checkpoints'],
    clean_previous_logs=training_config['clean_previous_logs'],
)
# ---------------------------------------------

# 2 -- Create the agent and the environment
# 2.1 -- Initialize  RAY

# ray.init(num_cpus=config['num_workers'] + 1, local_mode=False)
# ---------------------------------------------

# 2.2 -- creating the trainer and the env

config = dqn.DQNConfig().update_from_dict(config)

agent = dqn.DQN(config=config, env=Furnace, logger_creator=custom_log_creator(training_config["logger_dir"]))
if training_config['resume_training']:
    agent.restore(training_config['restoring_dir'])
# ---------------------------------------------

# 3 -- Training and savings
for i in range(nr_training_rounds):
    result = agent.train()
    print(pretty_print(result))

    if i % checkpoint_interval == 0:
        checkpoint = agent.save(
            checkpoint_dir=training_config['checkpoint_dir'] + "/checkpoint_" + str(i),
        )
        print('checkpoint saved at', checkpoint)
