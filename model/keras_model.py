from __future__ import annotations

import tensorflow as tf
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch


class KerasQModel(TFModelV2):
    """Custom model for DQN."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        self.original_space = (
            obs_space.original_space
            if hasattr(
                obs_space,
                'original_space',
            )
            else obs_space
        )
        self.input_temperature = tf.keras.layers.Input(
            shape=self.original_space['temperature'].shape,
            name='temperature',
        )
        self.input_timestep = tf.keras.layers.Input(
            shape=self.original_space['timestep'].shape,
            name='timestep',
        )
        self.input_PF = tf.keras.layers.Input(
            shape=self.original_space['PF'].shape,
            name='PF',
        )
        # import pdb; pdb.set_trace()
        # the image processing part:

        # 128-3+1 = 126
        x = tf.keras.layers.SeparableConv2D(
            filters=4,
            kernel_size=(
                3,
                3,
            ),
            strides=(2, 2),
            padding='valid',
        )(self.input_PF)
        # print(x.shape)
        x = tf.keras.layers.SeparableConv2D(
            filters=8,
            kernel_size=(
                3,
                3,
            ),
            strides=(2, 2),
            padding='valid',
            activation='selu',
        )(x)
        # print(x.shape)
        x = tf.keras.layers.SeparableConv2D(
            filters=16,
            kernel_size=(
                3,
                3,
            ),
            strides=(2, 2),
            padding='valid',
            activation='selu',
        )(x)
        # print(x.shape)
        x = tf.keras.layers.SeparableConv2D(
            filters=32,
            kernel_size=(
                3,
                3,
            ),
            strides=(2, 2),
            padding='valid',
            activation='selu',
        )(x)
        x = tf.keras.layers.Flatten()(x)
        # print(x.shape)
        x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(x)
        embedded = tf.keras.layers.Concatenate()(
            [self.input_temperature, self.input_timestep, x],
        )

        y1 = embedded
        multiplier = 1

        neurons_lst = [
            128 * multiplier,
            64 * multiplier,
            32 * multiplier,
            16 * multiplier,
            multiplier * (action_space.n - 1),
        ]
        for counter, n in enumerate(neurons_lst):
            y1 = tf.keras.layers.Dense(
                n,
                name='embedded_to_dT_action_' + str(counter),
                activation=tf.nn.relu,
            )(y1)

        y2 = embedded
        neurons_lst = [32 * multiplier, 16 * multiplier, multiplier * 1]
        for counter, n in enumerate(neurons_lst):
            y2 = tf.keras.layers.Dense(
                n,
                name='embedded_to_stop_action_' + str(counter),
                activation=tf.nn.relu,
            )(y2)

        pre_layer_out = tf.keras.layers.Concatenate()([y1, y2])

        # neurons_lst = [2 * action_space.n, action_space.n]
        # for counter, n in enumerate(neurons_lst):
        # pre_layer_out = tf.keras.layers.Dense(
        # n,
        # name="combined_to_output" + str(counter),
        # activation=tf.nn.relu,)(pre_layer_out)

        pre_layer_out = tf.keras.layers.Dense(
            num_outputs,
            name='what' + str(counter),
            activation=tf.nn.relu,
        )(pre_layer_out)

        layer_out = pre_layer_out

        self.model = tf.keras.Model(
            [self.input_temperature, self.input_timestep, self.input_PF],
            layer_out,
        )
        self.model.summary()

    # Implement the core forward method.
    def forward(self, input_dict, state, seq_lens):

        if SampleBatch.OBS in input_dict and 'obs_flat' in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS],
                self.obs_space,
                'tf',
            )

        inputs = {
            'temperature': orig_obs['temperature'],
            'timestep': orig_obs['timestep'],
            'PF': orig_obs['PF'],
        }
        model_out = self.model(inputs)

        return model_out, state
