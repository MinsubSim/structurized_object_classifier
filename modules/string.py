import tensorflow as tf
from . import UnitModule



class BasicCNNModule(UnitModule):
    def __init__(self, num_units, dropout_val, embedding_size, filter_size):
        super().__init__(num_units=num_units, dropout_val=dropout_val)
        self.embedding_size = embedding_size
        self.filter_size = filter_size

    def build(self, feature, input_tensor, state):
        char_depth = feature.char_depth
        string_length = feature.string_length
        string_tensor = input_tensor[1:]

        emb_table = tf.get_variable('embedding',
                                    shape=(char_depth, self.embedding_size),
                                    initializer=tf.glorot_uniform_initializer())
        filter_shape = (self.filter_size, self.embedding_size, 1, self.num_units)
        filter_var = tf.get_variable('filter',
                                     shape=filter_shape,
                                     initializer=tf.glorot_normal_initializer())
        bias = tf.get_variable('bias',
                               shape=(self.num_units,),
                               initializer=tf.constant_initializer(0.1))
        embd_input = tf.nn.embedding_lookup(emb_table, string_tensor)
        embd_input = tf.expand_dims(embd_input, -1)

        conv = tf.nn.conv2d(embd_input,
                            filter_var,
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name='conv')
        biased = tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu')
        pooled = tf.nn.max_pool(biased,
                                ksize=[1, string_length - self.filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name='pool')
        pooled = tf.nn.dropout(pooled, self.dropout_var)
        result_vector = tf.squeeze(pooled, [1, 2])
        return result_vector, state


class WordLayerModule(UnitModule):
    def build(self, feature, input_tensor, state):
        string_tensor = tf.cast(input_tensor[1:], tf.int32)
        word_layer = tf.get_variable('word_layer',
                                      shape=(feature.vector_depth, self.num_units, self.num_units),
                                      initializer=tf.glorot_uniform_initializer())

        bias = tf.get_variable('bias',
                               shape=(1, self.num_units,),
                               initializer=tf.glorot_uniform_initializer())

        reduce_prod = tf.reduce_prod(tf.gather(word_layer, string_tensor), axis=0)
        return tf.matmul(state, reduce_prod) + bias
