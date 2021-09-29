import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, \
Dense, GRU, Input, Concatenate, Softmax, Lambda
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class SimilarityLayer(tf.keras.layers.Layer):
    def __init__(self, MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, **kwargs):
        super(SimilarityLayer, self).__init__(**kwargs)
        self.MAX_QUESTION_LENGTH = MAX_QUESTION_LENGTH
        self.MAX_CONTEXT_LENGTH = MAX_CONTEXT_LENGTH

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[0][-1 ] *3 ,1), trainable=True, name='sim_w')

    def call(self, inputs):
        H, U = inputs
        H_dim_repeat = [1, 1, self.MAX_QUESTION_LENGTH, 1]
        U_dim_repeat = [1, self.MAX_CONTEXT_LENGTH, 1, 1]
        repeated_H = K.tile(K.expand_dims(H, axis=2), H_dim_repeat)
        repeated_U = K.tile(K.expand_dims(U, axis=1), U_dim_repeat)
        element_wise_multiply = repeated_H * repeated_U
        concatenated_tensor = K.concatenate(
            [repeated_H, repeated_U, element_wise_multiply], axis=-1)
        dot_product = K.squeeze(K.dot(concatenated_tensor, self.w), axis=-1)
        return dot_product

    def get_config(self):
        config = super().get_config()
        # config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class C2Q(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(C2Q, self).__init__(**kwargs)

    def call(self, inputs):
        U, S = inputs
        a = Softmax(name='a')(S)
        U2 = K.expand_dims(U, axis=1)
        return K.sum(K.expand_dims(a, axis=-1 ) * U2, -2)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Q2C(tf.keras.layers.Layer):
    def __init__(self, MAX_CONTEXT_LENGTH, **kwargs):
        super(Q2C, self).__init__(**kwargs)
        self.MAX_CONTEXT_LENGTH = MAX_CONTEXT_LENGTH

    def call(self, inputs):
        H, S = inputs
        b = Softmax(name='b')(K.max(S, axis=-1))
        b = tf.expand_dims(b, -1)
        # h_ = b*H
        h_ = K.sum( b *H, -2)
        h_2 = K.expand_dims(h_, 1)
        return K.tile(h_2, [1, self.MAX_CONTEXT_LENGTH, 1])

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MergeG(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MergeG, self).__init__(**kwargs)

    def call(self, inputs):
        H, U_, H_ = inputs
        HU_ = H* U_
        HH_ = H * H_
        return K.concatenate([H, U_, HU_, HH_])

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Prediction(layers.Layer):
    def __init__(self, **kwargs):
        super(Prediction, self).__init__(**kwargs)

    def call(self, inputs):
        s, e = inputs
        s = tf.expand_dims(s, axis=2)
        e = tf.expand_dims(e, axis=1)
        outer = tf.matmul(s, e)
        outer = tf.linalg.band_part(outer, 0, 15)
        return outer

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_model(MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, EMBEDDING_DIM, embedding_matrix, char_embedding_matrix):
    VOCAB_SIZE = embedding_matrix.shape[0]
    CHAR_EMBEDDING_DIM = 50
    UNITS = int(EMBEDDING_DIM/2)

    # inputs
    input_question = Input(shape=(MAX_QUESTION_LENGTH,), dtype='int32', name='question')
    input_context = Input(shape=(MAX_CONTEXT_LENGTH,), dtype='int32', name='context')

    # encodings
    question_encoding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, trainable=False,
                                  input_length=MAX_QUESTION_LENGTH, mask_zero=True,
                                  embeddings_initializer=Constant(embedding_matrix),
                                  name='q_encoding')(input_question)

    paragraph_encoding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, trainable=False,
                                   input_length=MAX_CONTEXT_LENGTH, mask_zero=True,
                                   embeddings_initializer=Constant(embedding_matrix),
                                   name='p_encoding')(input_context)


    char_question_encoding = Embedding(VOCAB_SIZE, CHAR_EMBEDDING_DIM, trainable=False,
                                       input_length=MAX_QUESTION_LENGTH, mask_zero=True,
                                       embeddings_initializer=Constant(char_embedding_matrix),
                                       name='char_q_encoding')(input_question)

    char_paragraph_encoding = Embedding(VOCAB_SIZE, CHAR_EMBEDDING_DIM, trainable=False,
                                        input_length=MAX_CONTEXT_LENGTH, mask_zero=True,
                                        embeddings_initializer=Constant(char_embedding_matrix),
                                        name='char_p_encoding')(input_context)

    p = Concatenate(-1, name='concat_p')([paragraph_encoding, char_paragraph_encoding])
    p2 = Dense(2 * UNITS, 'relu', name='dense_p')(p)

    q = Concatenate(-1, name='concat_q')([question_encoding, char_question_encoding])
    q2 = Dense(2 * UNITS, 'relu', name='dense_q')(q)

    # P rnn
    H = Bidirectional(LSTM(UNITS, return_sequences=True, dropout=0.3, name='H'), name='biH')(p2)

    # Q rnn
    U = Bidirectional(LSTM(UNITS, return_sequences=True, dropout=0.3, name='U'), name='biU')(q2)

    S = SimilarityLayer(MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, name='S')([H, U])
    U_ = C2Q(name='C2Q')([U, S])
    H_ = Q2C(MAX_CONTEXT_LENGTH, name='Q2C')([H, S])

    G = MergeG(name='G')([H, U_, H_])

    M = Bidirectional(GRU(UNITS, return_sequences=True, dropout=0.3), name='M')(G)

    GM = Concatenate(name='GM')([G, M])
    start = tf.keras.layers.TimeDistributed(Dense(1, name='dense_s'), name='td_s')(GM)
    start = Softmax(name='start_')(tf.squeeze(start, -1))

    M2 = Bidirectional(GRU(UNITS, return_sequences=True, dropout=0.3), name='M2')(M)
    GM2 = Concatenate(name='GM2')([G, M2])
    end = tf.keras.layers.TimeDistributed(Dense(1, name='dense_e'), name='td_e')(GM2)
    end = Softmax(name='end_')(tf.squeeze(end, -1))

    outer = Prediction(name='prediction')([start, end])
    start_pos = Lambda(lambda x: tf.reduce_max(x, axis=2), name='start')(outer)
    end_pos = Lambda(lambda x: tf.reduce_max(x, axis=1), name='end')(outer)

    model = Model([input_context, input_question],
                  # char_input_question, char_input_context],
                  [start_pos, end_pos])
    return model