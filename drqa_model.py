import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, GRU, Input, Concatenate, Softmax, \
    Attention, Dot, Lambda, LSTM
from tensorflow.keras.models import Model

UNITS = 100

class AlignedQ(layers.Layer):
    def __init__(self, units=UNITS * 2):
        super(AlignedQ, self).__init__(name='AlignedQ')
        self.units = units
        self.alpha1 = Dense(self.units, 'relu')
        self.alpha2 = Dense(self.units, 'relu')

    def call(self, inputs, **kwargs):
        p, q = inputs
        alpha_p = self.alpha1(p)
        alpha_q = self.alpha2(q)
        return Attention()([alpha_p, q, alpha_q])

    def get_config(self):
        config = super(AlignedQ, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class WeightedSum(layers.Layer):
    def __init__(self):
        super(WeightedSum, self).__init__(name='WeightedSum')
        self.w = Dense(1, 'softmax', use_bias=False)

    def call(self, inputs, **kwargs):
        q = inputs
        b = self.w(q)
        return tf.math.reduce_sum(b * q, 1)
        
    def get_config(self):
        config = super(WeightedSum, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SimilarityS(layers.Layer):
    def __init__(self, units=UNITS * 2):
        super(SimilarityS, self).__init__(name='start_sim')
        self.units = units
        self.WS = Dense(self.units)

    def call(self, inputs, **kwargs):
        p, q = inputs
        WSq = tf.expand_dims(self.WS(q), 1)
        pWSq = Dot(-1)([p, WSq])
        return tf.squeeze(pWSq, -1)

    def get_config(self):
        config = super(SimilarityS, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SimilarityE(layers.Layer):
    def __init__(self, units=UNITS * 2):
        super(SimilarityE, self).__init__(name='end_sim')
        self.units = units
        self.WE = Dense(units)

    def call(self, inputs, **kwargs):
        p, q = inputs
        WEq = tf.expand_dims(self.WE(q), 1)
        pWEq = Dot(-1)([p, WEq])
        return tf.squeeze(pWEq, -1)

    def get_config(self):
        config = super(SimilarityE, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Prediction(layers.Layer):
    def __init__(self):
        super(Prediction, self).__init__(name='prediction')

    def call(self, inputs, **kwargs):
        s, e = inputs
        s = tf.expand_dims(s, axis=2)
        e = tf.expand_dims(e, axis=1)
        outer = tf.matmul(s, e)
        outer = tf.linalg.band_part(outer, 0, 15)
        return outer
        
    def get_config(self):
        config = super(Prediction, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_model(max_question_length, max_context_length, embedding_dim, embedding_matrix, pos_embedding_matrix,
                ner_embedding_matrix):
    # inputs
    VOCAB_SIZE = embedding_matrix.shape[0]
    units = 100
    input_question = Input(shape=(max_question_length,), dtype='int32', name='question')
    input_context = Input(shape=(max_context_length,), dtype='int32', name='context')
    input_em = Input(shape=(max_context_length, 3), dtype='float32', name='em')
    input_pos = Input(shape=(max_context_length,), dtype='int32', name='pos')
    input_ner = Input(shape=(max_context_length,), dtype='int32', name='ner')
    input_tf = Input(shape=(max_context_length, 1), dtype='float32', name='tf')

    # encodings
    question_encoding = Embedding(VOCAB_SIZE, embedding_dim, trainable=False,
                                  input_length=max_question_length, mask_zero=True,
                                  embeddings_initializer=Constant(embedding_matrix),
                                  name='q_encoding')(input_question)

    paragraph_encoding = Embedding(VOCAB_SIZE, embedding_dim, trainable=False,
                                   input_length=max_context_length, mask_zero=True,
                                   embeddings_initializer=Constant(embedding_matrix),
                                   name='p_encoding')(input_context)

    pos_encoding = Embedding(pos_embedding_matrix.shape[0], pos_embedding_matrix.shape[1], trainable=False,
                             input_length=max_context_length, mask_zero=True,
                             embeddings_initializer=Constant(pos_embedding_matrix),
                             name='pos_encoding')(input_pos)

    ner_encoding = Embedding(ner_embedding_matrix.shape[0], ner_embedding_matrix.shape[1], trainable=False,
                             input_length=max_context_length, mask_zero=True,
                             embeddings_initializer=Constant(ner_embedding_matrix),
                             name='ner_encoding')(input_ner)

    q0 = Bidirectional(GRU(units, return_sequences=True, dropout=0.5), name='q0')(question_encoding)
    p0 = Bidirectional(GRU(units, return_sequences=True, dropout=0.5), name='p0')(paragraph_encoding)

    aligned_q = AlignedQ()([p0, q0])

    p00 = Bidirectional(GRU(units, return_sequences=True, dropout=0.5), name='p00')(Concatenate()([aligned_q, p0]))

    # input for P rnn
    concat = Concatenate(axis=-1, name='concat')([p00, pos_encoding, ner_encoding, input_em, input_tf])

    # P rnn
    p = Bidirectional(GRU(units, return_sequences=True, dropout=0.5), name='p1')(concat)
    #p = Bidirectional(GRU(units, return_sequences=True), name='p2')(p)

    # Q rnn
    q = Bidirectional(GRU(units, return_sequences=True, dropout=0.5), name='q1')(q0)  # (question_encoding)
    #q = Bidirectional(GRU(units, return_sequences=True), name='q2')(q)

    # weighted sum q = sum(b*q), b is the weight vector
    q2 = WeightedSum()(q)

    # start end similarities
    sim_s = SimilarityS()([p, q2])
    sim_e = SimilarityE()([p, q2])

    # start end probabilities
    start = Softmax(name='start_prob')(sim_s)
    end = Softmax(name='end_prob')(sim_e)

    # start end indices (max distance 15)
    outer = Prediction()([start, end])
    start_pos = Lambda(lambda x: tf.reduce_max(x, axis=2), name='start')(outer)
    end_pos = Lambda(lambda x: tf.reduce_max(x, axis=1), name='end')(outer)

    model = Model([input_context, input_question, input_em, input_pos, input_ner, input_tf],
                  [start_pos, end_pos])
    return model
