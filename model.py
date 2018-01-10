from cell import *


class SOCModel:
  def __init__(self, struct, label_size, dropout_prob, learning_rate):
    self.struct = struct
    self.data_stack = []
    self.dropout_prob = dropout_prob
    self.dropout_var = tf.placeholder(dtype=tf.float32)
    self.label_size = label_size

    self.input_tensor = [tf.placeholder(shape=(None,)+s, dtype=tf.int32) for s in self.struct.tensor_shape]
    
    output = self.struct.model(self.input_tensor , self.dropout_var)

    '''
    self.logit = tf.contrib.layers.fully_connected(inputs=output, \
                                               num_outputs=label_size)
    '''
    W = tf.get_variable('W',
                        shape=(self.struct.vector_size, label_size),
                        initializer=tf.glorot_normal_initializer())
    b = tf.get_variable('b',
                        shape=(label_size,),
                        initializer=tf.glorot_normal_initializer())
    self.logit = tf.matmul(output, W) + b
    self.label_tensor = tf.placeholder(shape=[None], dtype=tf.int32)
    label_tensor_onehot = tf.one_hot(self.label_tensor, label_size)

    self.cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=label_tensor_onehot,
                                                         label_smoothing=0.3,
                                                         logits=self.logit)

    self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           ).minimize(self.cross_entropy)
    prediction = tf.argmax(tf.nn.softmax(self.logit), 1)
    correct_predictions = tf.equal(prediction, tf.cast(self.label_tensor, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    self.eval_set = [self.train_op, self.logit, self.cross_entropy, self.accuracy, prediction]

  def insert(self, objects, labels, metas):
    for obj, label, meta in zip(objects, labels, metas):
      res = self.struct.transform(obj)
      self.data_stack.append((res, label, meta))

  def batch(self, batch_size):
    input_data = [[] for _ in self.input_tensor]
    
    label_data = []
    meta_data = []
    for dat, label, meta in self.data_stack[:batch_size]:
      for i, d in enumerate(dat):
        input_data[i].append(d)
      label_data.append(label)
      meta_data.append(meta)
    del self.data_stack[:batch_size]
    return input_data, label_data, meta_data
