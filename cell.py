import tensorflow as tf


class SOCCell(tf.contrib.rnn.RNNCell):
  def __init__(self, vector_size):
    super(SOCCell, self).__init__()
    self.vector_size = vector_size
    self.dropout_var = None

  def __call__(self, input_tensor, state):
    model_o = self.model(input_tensor, self.dropout_var)
    return model_o, state

  def model(self, input_tensor, dropout_var):
    pass

  def transform(self, obj):
    return ([int(obj or 0)],)

  @property
  def state_size(self):
      return self.vector_size

  @property
  def output_size(self):
      return self.vector_size

####
class SOCListCell(SOCCell):
  def __init__(self,
               vector_size,
               list_length,
               elem,
               ):
    super(SOCListCell, self).__init__(vector_size)
    self.elem = elem
    self.tensor_shape = [(list_length, ) + s for s in self.elem.tensor_shape] + [()]
    self.list_length = list_length

  def model(self, input_tensor, dropout_var):
    with tf.name_scope("ListStruct"):
      cells = [self.elem]
      for i in range(2):
          cell = tf.contrib.rnn.LSTMCell(num_units=self.vector_size)
          if i == 0:
              # Add attention wrapper to first layer.
              cell = tf.contrib.rnn.AttentionCellWrapper(cell, 32, state_is_tuple=True)
          cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_var)
          cells.append(cell)
      
      sequence_length = input_tensor[-1]
      data_tensor = input_tensor[:-1]
    
      elem_cell = tf.contrib.rnn.MultiRNNCell(cells)
      self.elem.dropout_var = dropout_var
      output, state = tf.nn.dynamic_rnn(elem_cell,
                                        data_tensor,
                                        sequence_length=sequence_length,
                                        dtype=tf.float32)
      
      batch_range = tf.range(tf.shape(output)[0])
      indices = tf.stack([batch_range, sequence_length-1], axis=1)
      res = tf.gather_nd(output, indices)
      self.test = []
      return res

  def transform(self, obj):
    res = [[] for _ in self.tensor_shape]
    for o in obj[-self.list_length:]:
      dat = self.elem.transform(o)
      for i, t in enumerate(dat):
        res[i].append(t)
    dat = self.elem.transform(None)
    for i, t in enumerate(dat):
      res[i] += [t] * (self.list_length - len(obj))
    res[-1] = min(len(obj), self.list_length)
    return res

####
class SOCDictCell(SOCCell):
  def __init__(self,
               vector_size,
               struct,
               ):
    super(SOCDictCell, self).__init__(vector_size)
    key_list = sorted(struct.keys())
    tensor_shape = []
    tensor_map = {}
    for k in key_list:
      tensor_map[k] = []
      for shape in struct[k].tensor_shape:
        tensor_map[k].append(len(tensor_shape))
        tensor_shape.append(shape)

    self.struct = struct
    self.key_list = key_list
    self.tensor_shape = tensor_shape
    self.tensor_map = tensor_map

  def model(self, input_tensor, dropout_var):
    with tf.name_scope("DictStruct"):
      all_o = []
      for k in self.key_list:
        sub_input = [input_tensor[i] for i in self.tensor_map[k]]
        out = self.struct[k].model(sub_input, dropout_var)
        all_o.append(out)
      merged_i = tf.concat(all_o, axis=1)
      merged_o = tf.contrib.layers.fully_connected(inputs=merged_i, \
                                                   num_outputs=self.vector_size)
      return merged_o

  def transform(self, obj):
    obj = obj or {}
    res = [None] * len(self.tensor_shape)
    for key in self.key_list:
      sub_input = self.struct[key].transform(obj.get(key, None))
      for i, t in zip(self.tensor_map[key], sub_input):
        res[i] = t
    return res

####
class SOCIndexCell(SOCCell):
  def __init__(self,
               vector_size,
               ):
    super(SOCIndexCell, self).__init__(vector_size)
    self.tensor_shape = [(1,)]

  def model(self, input_tensor, dropout_var):
    with tf.name_scope("IndexStruct"):
      onehot_o = tf.one_hot(tf.squeeze(input_tensor[0], [-1]), self.vector_size+1)
      return onehot_o

  def transform(self, obj):
    if obj:
      obj = int(obj) + 1
    else:
      obj = 0
    return ([obj],)

####
class SOCFigureCell(SOCCell):
  def __init__(self,
               vector_size,
               ):
    super(SOCFigureCell, self).__init__(vector_size)
    self.tensor_shape = [(1,)]

  def model(self, input_tensor, dropout_var):
    with tf.name_scope("FigureStruct"):
      expand_var = tf.get_variable('expand',
                                   shape=(1, self.vector_size),
                                   initializer=tf.glorot_normal_initializer())
      expanded_o = tf.matmul(tf.cast(input_tensor[0], tf.float32), expand_var)
      return expanded_o

####
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

    self.tensor_shape = [(string_length, )]
    self.filter_size = filter_size
    self.embedding_size = embedding_size

  def model(self, input_tensor, dropout_var):
    with tf.name_scope("StringStruct"):
      emb_table = tf.get_variable('embedding',
                                  shape=(self.char_depth, self.embedding_size),
                                  initializer=tf.glorot_uniform_initializer())
      filter_shape = (self.filter_size, self.embedding_size, 1, self.vector_size)
      filter_var = tf.get_variable('filter',
                                   shape=filter_shape,
                                   initializer=tf.glorot_normal_initializer())
      bias = tf.get_variable('bias',
                             shape=(self.vector_size),
                             initializer=tf.constant_initializer(0.1))

      # Convolution Layer
      embd_input = tf.nn.embedding_lookup(emb_table, input_tensor[0])
      embd_input = tf.expand_dims(embd_input, -1)

      # Convolution Layer
      conv = tf.nn.conv2d(embd_input,
                          filter_var,
                          strides=[1, 1, 1, 1],
                          padding='VALID',
                          name='conv')
      # Apply nonlinearity
      biased= tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu')
      # Max-pooling over the outputs
      pooled = tf.nn.max_pool(biased,
                              ksize=[1, self.string_length - self.filter_size + 1, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding='VALID',
                              name='pool')
      pooled = tf.nn.dropout(pooled, dropout_var)

      return tf.squeeze(pooled, [1, 2])

  def transform(self, obj):
    obj = obj or ''
    return [self.string_encode(obj[-self.string_length:], self.string_length)]

  def string_encode(self, string, length):
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
