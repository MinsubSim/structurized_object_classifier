from units import *


class SOCModel:
  def __init__(self, struct, label_size, dropout_prob, learning_rate, num_gpus):
    self.struct = struct
    self.data_stack = []
    self.dropout_prob = dropout_prob
    self.dropout_var = tf.placeholder(dtype=tf.float32)
    self.label_size = label_size
    self.num_gpus = num_gpus

    input_tensor = [tf.placeholder(shape=(None,)+s, dtype=tf.int32) for s in self.struct.tensor_shape]
    self.input_tensors = []
    self.label_tensors = []

    # Calculate the gradients for each model tower.
    loss_list = []
    pred_list = []
    acc_list = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(self.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('soc_%d' % (i)) as scope:
            input_tensor = [tf.placeholder(shape=(None,)+s, dtype=tf.int32) for s in self.struct.tensor_shape]
            label_tensor = tf.placeholder(shape=[None], dtype=tf.int32)
            self.input_tensors.append(input_tensor)
            self.label_tensors.append(label_tensor)
            loss, pred = self.tower_loss(scope, input_tensor, label_tensor)
            loss_list.append(loss)
            pred_list.append(pred)
            correct_predictions = tf.equal(pred, tf.cast(label_tensor, tf.int64))
            accuracy = tf.cast(correct_predictions, tf.float32)
            acc_list.append(accuracy)
            tf.get_variable_scope().reuse_variables()
    self.loss_mean = tf.reduce_mean(loss_list)
    self.accuracy = tf.reduce_mean(acc_list)
    self.predictions = tf.concat(pred_list, axis=0)
    self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_mean)
    self.eval_set = [self.train_op, self.loss_mean, self.accuracy]

  def tower_loss(self, scope, input_tensor, label_tensor):
    output = self.struct.model(input_tensor, self.dropout_var)
    W = tf.get_variable('W',
                        shape=(self.struct.vector_size, self.label_size),
                        initializer=tf.glorot_normal_initializer())
    b = tf.get_variable('b',
                        shape=(self.label_size,),
                        initializer=tf.glorot_normal_initializer())
    logit = tf.matmul(output, W) + b
    label_tensor_onehot = tf.one_hot(label_tensor, self.label_size)
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=label_tensor_onehot,
                                                    label_smoothing=0.1,
                                                    logits=logit)
    prediction = tf.argmax(tf.nn.softmax(logit), 1)
    return cross_entropy, prediction

  def insert(self, objects, labels, metas):
    for obj, label, meta in zip(objects, labels, metas):
      res = self.struct.transform(obj)
      self.data_stack.append((res, label, meta))

  def batch(self, batch_size):
    input_data = [[] for _ in self.struct.tensor_shape]
    label_data = []
    meta_data = []
    for dat, label, meta in self.data_stack[:batch_size]:
      for i, d in enumerate(dat):
        input_data[i].append(d)
      label_data.append(label)
      meta_data.append(meta)
    del self.data_stack[:batch_size]
    return input_data, label_data, meta_data

  def train(self, sess, batch_size):
    input_data, label_data, meta_data = self.batch(batch_size)
    feed_dict = {
      self.dropout_var: self.dropout_prob,
    }
    meta_list = []
    for gpu_idx in range(self.num_gpus):
      input_data, label_data, meta_data = self.batch(batch_size)
      meta_list += meta_data
      feed_dict[self.label_tensors[gpu_idx]] = label_data
      for x, y in zip(self.input_tensors[gpu_idx], input_data):
        feed_dict[x] = y
    res = sess.run(self.eval_set, feed_dict=feed_dict)
    return res, meta_list

  def evaluate(self, sess, batch_size):
    input_data, label_data, meta_data = self.batch(batch_size)
    feed_dict = {
      self.dropout_var: 1.0,
    }
    meta_list = []
    for gpu_idx in range(self.num_gpus):
      input_data, label_data, meta_data = self.batch(batch_size)
      meta_list += meta_data
      feed_dict[self.label_tensors[gpu_idx]] = label_data
      for x, y in zip(self.input_tensors[gpu_idx], input_data):
        feed_dict[x] = y
    res = sess.run(self.eval_set, feed_dict=feed_dict)
    return res, meta_list
