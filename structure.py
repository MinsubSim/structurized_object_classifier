import tensorflow as tf



class BaseStruct(object):
  def __init__(self, vector_size):
    super(BaseStruct, self).__init__()

  def feed(self, input_data):
    pass

  def exp(self, batch_size, drop_out_val):
    pass



class ListStruct(BaseStruct):
  def __init__(self,
               elem,
               vector_siz,
               list_length,
               ):
    super(ListStruct, self).__init__(vector_size=vector_size)
    self.elem = elem
    self.vector_size = vector_size

    cell = tf.contrib.rnn.BasicLSTMCell(self.vector_size)
    outputs, state = tf.nn.dynamic_rnn(cell, input_tensors
                                       dtype=tf.float32)
    self.output_tensor = outputs[:, -1, :]
    self.seq_len_stack = []

  def feed(self, input_data):
    self.seq_len_stack.append(len(input_data))
    for elem_data in input_data:
      self.elem.feed(elem_data)



class DictStruct(BaseStruct):
  def __init__(self,
               shape,
               vector_size,
               ):
    super(DictStruct, self).__init__()
    self.shape = shape
    self.vector_size = vector_size

    with tf.name_scope("DictStruct"):
      self.key_list = sorted(self.shape.keys())
      self.input_tensor = tf.concat([self.shape[key].model() for key in key_list])
      self.output_tensor = tf.contrib.layers.fully_connected(inputs=input_tensor,
                                                             num_outputs=self.vector_size)

  def feed(self, input_data):
    for key in key_list:
      self.shape[key].feed(input_data[key])



class IndexStruct(BaseStruct):
  def __init__(self,
               vector_size,
               ):
    super(IndexStruct, self).__init__()
    self.vector_size = vector_size
    with tf.name_scope("IndexStruct"):
      self.input_tensor = tf.placeholder(tf.float32, [None])
    self.output_tensor = tf.one_hot(input=self.input_tensor, depth=self.vector_size)
    self.data_stack = []

  def feed(self, input_data):
    self.data_stack.append(input_data)



class FigureStruct(BaseStruct):
  def __init__(self,
               vector_size,
               ):
    super(FigureStruct, self).__init__()
    with tf.name_scope("FigureStruct"):
      self.input_tensor = tf.placeholder(tf.float32, [None])
      self.expand_var = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                    name='expand')
    self.output_tensor = tf.matmul(self.input_tensor, self.expand_var)
    self.data_stack = []

  def feed(self, input_data):
    self.data_stack.append(input_data)


class StringStruct(BaseStruct):
  def __init__(self,
               vector_size,
               string_length,
               char_depth,
               filter_size,
               embedding_size,
               dropout_prob,
               ):
    super(FigureStruct, self).__init__()
    self.vector_size = vector_size
    self.string_length = string_length
    self.char_depth = char_depth

    self,filter_size = filter_size
    self.embedding_size = embedding_size
    self.dropout_prob = dropout_prob

    with tf.name_scope("StringStruct"):
      self.input_tensor = tf.placeholder(tf.float32, [None, string_length], name="input")
      self.dropout_var = tf.placeholder(tf.float32, name="dropout_keep_prob")
      # Convolution Layer
      emb_table = tf.Variable(tf.random_uniform([self.char_depth, self.embedding_size],
                                                -1.0, 1.0),
                              name='embedding')
      embd_input = tf.nn.embedding_lookup(emb_table, self.input_tensor)
      embd_input = tf.expand_dims(embd_input, -1)

      # Convolution Layer
      filter_shape = [self.filter_size, self.embedding_size, 1, self.vector_size]
      filter_var = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filter')
      conv = tf.nn.conv2d(embd_input,
                          filter_var,
                          strides=[1, 1, 1, 1],
                          padding='VALID',
                          name='conv')
      # Apply nonlinearity
      bias = tf.Variable(tf.constant(0.1, shape=[self.vector_size]), name='bias')
      biased= tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu')
      # Max-pooling over the outputs
      pooled = tf.nn.max_pool(biased,
                              ksize=[1, self.string_length - self.filter_size + 1, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding='VALID',
                              name='pool')

    self.output_tensor = tf.nn.dropout(pooled, self.dropout_var)
    self.data_stack = []

  def feed(self, input_data):
    v_dat = vector_encode(input_data, string_length)
    self.data_stack.append(v_dat)
