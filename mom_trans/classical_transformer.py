import tensorflow as tf
from tensorflow import keras
import gc
import numpy as np

from mom_trans.deep_momentum_network import DeepMomentumNetworkModel, SharpeLoss
from settings.hp_grid import (
    HP_DROPOUT_RATE,
    HP_HIDDEN_LAYER_SIZE,
    HP_LEARNING_RATE,
    HP_MAX_GRADIENT_NORM,
    HP_MINIBATCH_SIZE,
)

# the position encoding
class Time2Vector(keras.layers.Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        # non-periodic representation, linear
        self.weights_linear = self.add_weight(name='weight_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)
        # periodic representation, sine wave
        self.weights_periodic = self.add_weight(name='weight_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

    def call(self, x):
        # calc the average of the feature values
        x = tf.math.reduce_mean(x, axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)
        return tf.concat([time_linear, time_periodic], axis=-1)

class SingleAttention(keras.layers.Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = keras.layers.Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.key = keras.layers.Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.value = keras.layers.Dense(self.d_v, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq), the Q,K,V in transformer
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out

class MultiAttention(keras.layers.Layer):
    def __init__(self, d_k, d_v, n_heads, input_cols):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()
        self.input_cols = input_cols

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))
        # self.input_cols + 2 means I concat the time embedding on it
        self.linear = keras.layers.Dense(self.input_cols + 2, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout, input_cols, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout
        self.input_cols = input_cols

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads, self.input_cols)
        self.attn_dropout = keras.layers.Dropout(self.dropout_rate)
        self.attn_normalize = keras.layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = keras.layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        self.ff_conv1D_2 = keras.layers.Conv1D(filters=self.input_cols+2, kernel_size=1) # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
        self.ff_dropout = keras.layers.Dropout(self.dropout_rate)
        self.ff_normalize = keras.layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)    

    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer

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

        time_embedding = Time2Vector(self.time_steps)
        d_k = 64
        d_v = 64
        n_heads = 8
        ff_dim = hidden_layer_size

        attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, dropout_rate, self.input_size)
        attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, dropout_rate, self.input_size)
        attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, dropout_rate, self.input_size)

        in_seq = keras.Input((self.time_steps, self.input_size))
        x = time_embedding(in_seq)
        x = keras.layers.Concatenate(axis=-1)([in_seq, x])

        print("===keras.layers.Concatenate:{}".format(x))

        x = attn_layer1((x, x, x))
        x = attn_layer2((x, x, x))
        x = attn_layer3((x, x, x))
        
        print("===attn_layer3:{}".format(x))


        x = keras.layers.Dense(hidden_layer_size, activation='relu')(x)
        x = keras.layers.Dropout(dropout_rate)(x)

        #x = tf.expand_dims(x, axis=-1)

        print("======x:{}".format(x))

        out = tf.keras.layers.TimeDistributed(
                keras.layers.Dense(self.output_size, activation=tf.nn.tanh, kernel_constraint=keras.constraints.max_norm(3))
              )(x)

        print("======out:{}".format(out))

        #out = keras.layers.Dense(self.output_size, activation=tf.nn.tanh)(x)


        model = keras.Model(inputs=in_seq, outputs=out)

        adam = keras.optimizers.Adam(lr=learning_rate, clipnorm=max_gradient_norm)
        sharpe_loss = SharpeLoss(self.output_size).call

        model.compile(
            loss=sharpe_loss,
            optimizer=adam,
            sample_weight_mode="temporal",
        )
        return model
