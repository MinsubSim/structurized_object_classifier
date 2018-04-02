import tensorflow as tf
import numpy as np

#TODO: 이게 구지 필요한가
#TODO: 이거까지 예제로 만들자
#TODO: 데이터 뽑는건 다른 라이브러리 pandas같은걸 사용


class SOCModel:
  def __init__(self, struct, label_size, learning_rate, device=['/cpu:0'], loss_func='softmax'):
    self.struct = struct
    self.data_stack = []
    self.dropout_var = tf.placeholder(dtype=tf.float32)
    self.label_size = label_size
    self.num_gpus = num_gpus
    self.loss_func = loss_func

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
            input_tensor = [tf.placeholder(shape=(None,)+s['shape'], dtype=s['dtype']) for s in self.struct.tensor_shape]
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
    self.eval_set = [self.train_op, self.loss_mean, self.accuracy, self.predictions]

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
    if self.loss_func == 'sigmoid':
      cross_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_tensor_onehot,
                                                      label_smoothing=0.1,
                                                      logits=logit)
    else:
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
    input_data, label_data, meta_data = zip(*self.data_stack[:batch_size])
    del self.data_stack[:batch_size]
    input_data = zip(*input_data)
    return input_data, label_data, meta_data

  def run(self, sess, batch_size, dropout_prob):
    feed_dict = {
      self.dropout_var: dropout_prob,
    }
    meta_list = []
    label_list = []
    for gpu_idx in range(self.num_gpus):
      input_data, label_data, meta_data = self.batch(batch_size)
      meta_list += meta_data
      label_list += label_data
      feed_dict[self.label_tensors[gpu_idx]] = label_data
      for x, y in zip(self.input_tensors[gpu_idx], input_data):
        feed_dict[x] = np.asarray(y, x.dtype.as_numpy_dtype)
    res = sess.run(self.eval_set, feed_dict=feed_dict)
    return res, label_list, meta_list

  def train(self, sess, batch_size):
    return self.run(sess, batch_size, self.dropout_prob)

  def evaluate(self, sess, batch_size):
    return self.run(sess, batch_size, 1.0)
