import tensorflow as tf



class SOCCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self, vector_size):
    super(SOCCell, self).__init__()
    self.vector_size = vector_size

  def feed(self, input_data):
    return input_data

  def call(self, input_tensor, state):
    model_o = self.model(input_tensor)
    return model_o, state

  @property
  def state_size(self):
      return self.vector_size

  @property
  def output_size(self):
      return self.vector_size

class SOCModel:
  def __init__(self, struct, dropout_prob, label_size):
    self.struct = struct
    self.data_stack = []
    self.dropout_prob = dropout_prob
    self.dropout_var = tf.placeholder(dtype=tf.float32)
    self.label_size = label_size

    num_units = 100
    cells = [self.struct]
    for _ in range(2):
        cell = tf.nn.rnn_cell.GRUCell(num_units=num_units)
        #if not cells:
            # Add attention wrapper to first layer.
        #    cell = tf.contrib.rnn.AttentionCellWrapper(cell, 50, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_var)
        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)

    tensor_shape = [(None, None) + s for s in self.struct.tensor_shape]
    self.input_tensor = tuple(tf.placeholder(shape=s, dtype=tf.int32) for s in tensor_shape)

    output, _ = tf.nn.dynamic_rnn(cell,
                                  self.input_tensor,
                                  time_major=True,
                                  dtype=tf.float32)

    print('output', output)
    W = tf.Variable(tf.random_normal([self.struct.vector_size, label_size], stddev=0.35))
    output = tf.matmul(output, W)
    label_tensor_onehot = tf.one_hot(label_tensor, label_size)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=label_tensor_onehot,
                                                    logits=output,
                                                    label_smoothing=0.1)

    train_op = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

  def feed(self, input_data, label_data):
    for data, label in input_data:
      self.data_stack.append(self.struct.feed(data))

  def train(self, batch_size):
    feed_dict = {
      self.input_tensor: self.pop_batch_size,
      self.dropout_prob: 0.9,
    }

    output_res, cost_res, _ = sess.run([output, cross_entropy, train_op],
                                       feed_dict=feed_dict)
    return cost_res

  def evaluate(self, batch_size):
    feed_dict = {
      self.input_tensor: self.pop_batch_size,
      self.dropout_prob: 0.9,
    }

    output_res, cost_res, _ = sess.run([output, cross_entropy, train_op],
                                       feed_dict=feed_dict)
    return cost_res


class SOCDictCell(SOCCell):
  def __init__(self,
               struct,
               vector_size,
               ):
    super(SOCDictCell, self).__init__(vector_size)
    self.struct = struct
    self.key_list = sorted(struct.keys())
    self.tensor_shape = tuple(struct[k].tensor_shape for k in self.key_list)

  def model(self, input_tensor):
    with tf.name_scope("DictStruct"):
      all_o = []
      for d, k in zip(input_tensor, self.key_list):
        out = self.struct[k].model(d)
        all_o.append(out)
      merged_i = tf.concat(all_o, axis=1)
      merged_o = tf.contrib.layers.fully_connected(inputs=merged_i, \
                                                   num_outputs=self.vector_size)

      print(merged_o)
      return merged_o

  def feed(self, input_data):
    res = []
    for key in key_list:
      res.append(self.struct[key].feed(input_data.get(key, None)))
    return res


class SOCIndexCell(SOCCell):
  def __init__(self,
               vector_size,
               ):
    super(SOCIndexCell, self).__init__(vector_size)
    self.tensor_shape = (1,)

  def model(self, input_tensor):
    with tf.name_scope("IndexStruct"):
      onehot_o = tf.one_hot(tf.squeeze(input_tensor, [-1]), self.vector_size)
      return onehot_o


class SOCFigureCell(SOCCell):
  def __init__(self,
               vector_size,
               ):
    super(SOCFigureCell, self).__init__(vector_size)
    self.tensor_shape = (1,)
    self.expand_var = tf.Variable(tf.truncated_normal([1, self.vector_size], stddev=0.1), name='expand')

  def model(self, input_tensor):
    with tf.name_scope("FigureStruct"):
      expanded_o = tf.matmul(tf.cast(input_tensor, tf.float32), self.expand_var)
      return expanded_o


class SOCStringCell(SOCCell):
  def __init__(self,
               vector_size,
               string_length,
               char_depth,
               filter_size,
               embedding_size,
               ):
    super(SOCStringCell, self).__init__(vector_size)
    self.string_length = string_length
    self.char_depth = char_depth

    self.tensor_shape = (string_length, )
    self.filter_size = filter_size
    self.embedding_size = embedding_size
    self.emb_table = tf.Variable(tf.random_uniform([self.char_depth, self.embedding_size],
                                                   -1.0, 1.0),
                                                   name='embedding')
    filter_shape = [self.filter_size, self.embedding_size, 1, self.vector_size]
    self.filter_var = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filter')
    self.bias = tf.Variable(tf.constant(0.1, shape=[self.vector_size]), name='bias')

  def model(self, input_tensor):
    with tf.name_scope("StringStruct"):
      # Convolution Layer
      embd_input = tf.nn.embedding_lookup(self.emb_table, input_tensor)
      embd_input = tf.expand_dims(embd_input, -1)

      # Convolution Layer
      conv = tf.nn.conv2d(embd_input,
                          self.filter_var,
                          strides=[1, 1, 1, 1],
                          padding='VALID',
                          name='conv')
      # Apply nonlinearity
      biased= tf.nn.relu(tf.nn.bias_add(conv, self.bias), name='relu')
      # Max-pooling over the outputs
      pooled = tf.nn.max_pool(biased,
                              ksize=[1, self.string_length - self.filter_size + 1, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding='VALID',
                              name='pool')

      return tf.squeeze(pooled, [1, 2])

  def feed(self, input_data):
    return self.vector_encode(input_data, string_length)

  def vector_encode(self, string, length):
    MATCH_MAP = [(0x1100, 0x11FF), (0xAC00, 0xD7A4), (0x0000, 0x00FF)] # 11682
    STR_DEPTH2 = 11682
    r = []
    for c in string:
        t = 1
        for s, e in MATCH_MAP:
            if s <= ord(c) and ord(c) <= e:
                t += ord(c) - s
                r.append(t)
                break
            t += e - s
    if len(r) < length:
        r += [0 for _ in range(length - len(r))]
    return r
