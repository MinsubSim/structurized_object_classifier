class ExpansionModel(UnitModel):
  def struct(self, vector_size, input_tensor, dropout_var):
    with tf.name_scope("ExpansionModel"):
      expand_var = tf.get_variable('expand',
                                   shape=(1, vector_size),
                                   initializer=tf.glorot_normal_initializer())
      expanded_o = tf.matmul(tf.cast(input_tensor[0], tf.float32), expand_var)
      return expanded_o

class OneHotModel(UnitModel):
  def struct(self, vector_size, input_tensor, dropout_var):
    with tf.name_scope("OneHotModel"):
      onehot_o = tf.one_hot(tf.squeeze(input_tensor[0], [-1]), vector_size)
      return onehot_o
