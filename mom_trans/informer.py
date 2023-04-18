import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine.base_layer import Layer
from mom_trans.classical_transformer import TransformerDeepMomentumNetworkModel, MultiAttention
from math import sqrt
import math
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


class ProbMask():
    def __init__(self, B, H, L, index, scores):
        _mask = tf.ones((L, scores.shape[-1]))

        mask_a = tf.linalg.band_part(_mask, 0, -1)  # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(_mask, 0, 0)  # Diagonal matrix of 0s and 1s
        _mask = tf.cast(mask_a - mask_b, dtype=tf.float32)

        _mask_ex = tf.broadcast_to(_mask, [B, H, L, scores.shape[-1]])
        indicator = _mask_ex[tf.range(B)[:, None, None],
                    tf.range(H)[None, :, None],
                    index, :]
        self._mask = indicator.reshape(scores.shape)

    @property
    def mask(self):
        return self._mask

class ProbAttention(keras.layers.Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = tf.keras.layers.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        # B, H, L, E = K.shape
        # _, _, S, _ = Q.shape
        B, H, L, E = K.shape
        _, _, S, _ = Q.shape
        # print("============key:",K.shape,tf.shape(K))
        # print("============key:",Q.shape,tf.shape(Q))
        # print("=====:",B,H,L,E,S)
        E = 1

        # calculate the sampled Q_K
        K_expand = tf.broadcast_to(tf.expand_dims(K, -3), (B, H, S, L, E))

        indx_q_seq = tf.random.uniform((S,), maxval=L, dtype=tf.int32)
        indx_k_seq = tf.random.uniform((sample_k,), maxval=L, dtype=tf.int32)

        K_sample = tf.gather(K_expand, tf.range(S), axis=2)

        K_sample = tf.gather(K_sample, indx_q_seq, axis=2)
        K_sample = tf.gather(K_sample, indx_k_seq, axis=3)

        Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, -2), tf.einsum("...ij->...ji", K_sample)))
        # find the Top_k query with sparisty measurement
        M = tf.math.reduce_max(Q_K_sample, axis=-1) - tf.raw_ops.Div(x=tf.reduce_sum(Q_K_sample, axis=-1), y=L)
        M_top = tf.math.top_k(M, n_top, sorted=False)[1]
        batch_indexes = tf.tile(tf.range(Q.shape[0])[:, tf.newaxis, tf.newaxis], (1, Q.shape[1], n_top))
        head_indexes = tf.tile(tf.range(Q.shape[1])[tf.newaxis, :, tf.newaxis], (Q.shape[0], 1, n_top))

        M_top = tf.reshape(M_top, (1,8,25))
        idx = tf.stack(values=[batch_indexes, head_indexes, M_top], axis=-1)

        # use the reduced Q to calculate Q_K
        Q_reduce = tf.gather_nd(Q, idx)

        Q_K = tf.matmul(Q_reduce, tf.transpose(K, [0, 1, 3, 2]))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        V = tf.reshape(V,(B,H,L_V,1))
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = tf.reduce_sum(V, -2)
            # print("=================V_SUM:",[B, H, L_Q, V_sum.shape[-1]])
            # print("=================",V_sum.shape,tf.expand_dims(V_sum, -2).shape)
            contex = tf.identity(tf.broadcast_to(tf.expand_dims(V_sum, -2), [B, H, L_Q, V_sum.shape[-1]]))
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = tf.math.cumsum(V, axis=-1)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores)

            # scores.masked_fill_(attn_mask.mask, -np.inf)
            num = 3.4 * math.pow(10, 38)
            scores = (scores * attn_mask.mask) + (-((attn_mask.mask * num + num) - num))

        attn = tf.keras.activations.softmax(scores, axis=-1)  # nn.Softmax(dim=-1)(scores)
        batch_indexes = tf.tile(tf.range(V.shape[0])[:, tf.newaxis, tf.newaxis], (1, V.shape[1], index.shape[-1]))
        head_indexes = tf.tile(tf.range(V.shape[1])[tf.newaxis, :, tf.newaxis], (V.shape[0], 1, index.shape[-1]))

        idx = tf.stack(values=[batch_indexes, head_indexes, index], axis=-1)

        context_in = tf.tensor_scatter_nd_update(context_in, idx, tf.matmul(attn, V))

        return tf.convert_to_tensor(context_in)

    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        D = 1

        queries = tf.reshape(queries, (B, H, L, -1))
        keys = tf.reshape(keys, (B, H, S, -1))
        values = tf.reshape(values, (B, H, S, -1))

        U = self.factor * np.ceil(np.log(S)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()

        scores_top, index = self._prob_QK(queries, keys, u, U)
        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L)
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index, L)

        return context
    
class AttentionLayer(keras.layers.Layer):
    def __init__(self, d_k, d_v, n_heads, input_cols):
        super(AttentionLayer, self).__init__()

        d_keys = d_k
        d_values = d_v
        d_model = d_k * n_heads
        self.d_model = d_model
        # d_model = 512
        self.input_cols = input_cols


        self.inner_attention = ProbAttention(False, factor=5, attention_dropout=0)

        self.query_projection = tf.keras.layers.Dense(d_keys * n_heads)
        self.key_projection = tf.keras.layers.Dense(d_keys * n_heads)
        self.value_projection = tf.keras.layers.Dense(d_values * n_heads)
        self.out_projection = tf.keras.layers.Dense(d_model)
        self.n_heads = n_heads

    def build(self, input_shape):
        # print("===============input_shape:",input_shape)
        B, L, _ = input_shape[0]
        _, S, _ = input_shape[1]
        # _, B, L = input_shape[0]
        # _, _, S = input_shape[1]
        B = 1
        H = self.n_heads
        # print("===============0")
        self.queries = self.add_weight(shape=(B, L, H, self.d_model),
                                 initializer='random_normal',
                                 trainable=True)
        # print("===============1")
        self.keys = self.add_weight(shape=(B, S, H, self.d_model),
                                       initializer='random_normal',
                                       trainable=True)
        # print("===============2")
        self.values = self.add_weight(shape=(B, S, H, self.d_model),
                                       initializer='random_normal',
                                       trainable=True)
        # print("===============3")
        self.linear = keras.layers.Dense(self.input_cols + 2, input_shape=input_shape, 
                                         kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        B = 1
        # _, B, L = queries.shape
        # _, _, S = keys.shape
        H = self.n_heads

        self.queries = tf.reshape(self.query_projection(queries), (B, L, H, -1))
        self.keys = tf.reshape(self.key_projection(keys), (B, S, H, -1))
        self.values = tf.reshape(self.value_projection(values), (B, S, H, -1))


        out = tf.reshape(self.inner_attention([self.queries, self.keys, self.values], attn_mask=attn_mask), (B, L, -1))

        # return self.out_projection(out)
        return self.linear(out)


class InformerEncoder(keras.layers.Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout, input_cols, atten_type=False, **kwargs):
        super(InformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout
        self.input_cols = input_cols
        self.atten_type = atten_type

    def build(self, input_shape):
        if self.atten_type:
            self.attn_multi = AttentionLayer(self.d_k, self.d_v, self.n_heads, self.input_cols)
        else:
            self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads, self.input_cols)
        # self.attn_multi = AttentionLayer(self.d_k, self.d_v, self.n_heads, self.input_cols)

        self.attn_dropout = keras.layers.Dropout(self.dropout_rate)
        self.attn_normalize = keras.layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = keras.layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        self.ff_conv1D_2 = keras.layers.Conv1D(filters=self.input_cols+2, kernel_size=1) # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
        self.ff_dropout = keras.layers.Dropout(self.dropout_rate)
        self.ff_normalize = keras.layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
        self.activation = tf.keras.activations.relu

    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)
        # same meaning
        # x = x + self.dropout(self.attention(
        #     [x, x, x],
        #     attn_mask = attn_mask
        # ))
        # y = x = self.norm1(x)

        ff_layer = self.ff_conv1D_1(attn_layer) # conv+activation
        ff_layer = self.ff_dropout(ff_layer)
        # equal: y = self.dropout(self.activation(self.conv1(y)))
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        # equal: y = self.dropout(self.conv2(y))
        ff_layer = self.ff_normalize(attn_layer + ff_layer)
        # equal: self.norm2(x+y)
        return ff_layer
    

class InformerDeepMomentumNetworkModel(DeepMomentumNetworkModel):
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

        attn_layer1 = InformerEncoder(d_k, d_v, n_heads, ff_dim, dropout_rate, self.input_size)
        attn_layer2 = InformerEncoder(d_k, d_v, n_heads, ff_dim, dropout_rate, self.input_size)
        attn_layer3 = InformerEncoder(d_k, d_v, n_heads, ff_dim, dropout_rate, self.input_size)

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


