import tensorflow as tf
from . import UnitModule



class ExpansionModule(UnitModule):
  def build(self, unit, input_tensor, dropout_var):
    with tf.name_scope("ExpansionModule"):
      expand_var = tf.get_variable('expand',
                                   shape=(1, unit.vector_size),
                                   initializer=tf.glorot_normal_initializer())
      expanded_o = tf.matmul(tf.cast(input_tensor[0], tf.float32), expand_var)
      return expanded_o


class OneHotModule(UnitModule):
  def build(self, unit, input_tensor, dropout_var):
    with tf.name_scope("OneHotModule"):
      onehot_o = tf.one_hot(tf.squeeze(input_tensor[0], [-1]), unit.vector_size)
      return onehot_o
