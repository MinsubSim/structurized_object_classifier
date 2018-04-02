import tensorflow as tf
from . import UnitModule



class ExpansionModule(UnitModule):
    def build(self, feature, input_tensor, state):
        input_data = input_tensor[0]
        expand_var = tf.get_variable('expand',
                                     shape=(1, self.num_units),
                                     initializer=tf.glorot_normal_initializer())
        matmul_res = tf.matmul(tf.cast(input_data, tf.float32), expand_var)
        return tf.contrib.layers.dropout(inputs=matmul_res, keep_prob=feature.dropout_var), state


class OneHotModule(UnitModule):
    def build(self, feature, input_tensor, state):
        input_data = input_tensor[0]
        onehot_res = tf.one_hot(tf.squeeze(input_data, [-1]), feature.vector_depth)
        fcl_res = tf.contrib.layers.fully_connected(inputs=onehot_res,
                                                    num_outputs=self.num_units)
        return tf.contrib.layers.dropout(inputs=fcl_res, keep_prob=feature.dropout_var), state
