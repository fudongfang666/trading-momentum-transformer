import tensorflow as tf
from tensorflow import keras
import gc
import numpy as np

concat = keras.backend.concatenate
stack = keras.backend.stack
K = keras.backend
Add = keras.layers.Add
LayerNorm = keras.layers.LayerNormalization
Dense = keras.layers.Dense
Multiply = keras.layers.Multiply
Dropout = keras.layers.Dropout
Activation = keras.layers.Activation
Lambda = keras.layers.Lambda

from mom_trans.deep_momentum_network import DeepMomentumNetworkModel, SharpeLoss
from settings.hp_grid import (
    HP_DROPOUT_RATE,
    HP_HIDDEN_LAYER_SIZE,
    HP_LEARNING_RATE,
    HP_MAX_GRADIENT_NORM,
    HP_MINIBATCH_SIZE,
)

class TransformerDeepMomentumNetworkModel(DeepMomentumNetworkModel):
    def __init__(
        self, project_name, hp_directory, hp_minibatch_size=HP_MINIBATCH_SIZE, **params
    ):
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)

    def model_builder(self, hp):
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE)
        dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM)
        learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE)
        # input feature shape
        input = keras.Input((self.time_steps, self.input_size))
