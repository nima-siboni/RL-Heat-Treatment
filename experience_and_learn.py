import numpy as np
import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.schedules import Schedule, PiecewiseSchedule
from Furnace import Furnace
from model.keras_model import KerasQModel
from utils.utils import create_config, create_end_points, custom_log_creator, \
    create_or_clean_training_dirs, \
    load_training_config

# ---------------------------------------------
# 1 - Setting the configs
# 2 - Creating the agent in its env
# 3 - training and saving
# ---------------------------------------------

# 1.1 -- The config of the DDQN trainer and its env.

# 1.1.0 -- register the keras model
keras_q_model = KerasQModel
ModelCatalog.register_custom_model("keras_Q_model", keras_q_model)

# 1.1.1 -- the DQN configs

config = create_config(base_config=dqn.DEFAULT_CONFIG.copy(),
                       env_cfg_dir='./configs/',
                       agent_cfg_dir='./configs/')

# ---------------------------------------------

endpoints = create_end_points()

config["exploration_config"].update(
    {  # Further Configs for the Exploration class' constructor:
        "epsilon_schedule": PiecewiseSchedule(endpoints=endpoints, framework="tf",
                                              outside_value=0.01)
    }
)

# 1.2 -- Rounds of training and Check point parameters
# (if and how often to write the network)
training_config = load_training_config()
nr_training_rounds = training_config["nr_training_rounds"]

checkpoint_interval = training_config["checkpoint_interval"]

create_or_clean_training_dirs(checkpoint_dir=training_config["checkpoint_dir"],
                              logger_dir=training_config["logger_dir"],
                              clean_previous_checkpoints=training_config[
                                  "clean_previous_checkpoints"],
                              clean_previous_logs=training_config[
                                  "clean_previous_logs"])
# ---------------------------------------------

# 2 -- Create the agent and the environment
# 2.1 -- Initialize  RAY
ray.init(num_cpus=config["num_workers"] + 1)
# ---------------------------------------------

# 2.2 -- creating the trainer and the env
trainer = dqn.DQNTrainer(
    config=config,
    env=Furnace,
    logger_creator=custom_log_creator(training_config["logger_dir"]))

if training_config["resume_training"]:
    trainer.restore(training_config["restoring_dir"])
# ---------------------------------------------

# 3 -- Training and savings
for i in range(nr_training_rounds):
    result = trainer.train()
    print(pretty_print(result))

    if i % checkpoint_interval == 0:
        checkpoint = trainer.save(checkpoint_dir=training_config["checkpoint_dir"])
        print("checkpoint saved at", checkpoint)
