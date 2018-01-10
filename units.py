import tensorflow as tf


class SOCUnit(tf.contrib.rnn.RNNCell):

  class UnitShape:
    def __init__(self, optional):
      self.tensor_info = []
      if optional:
        self.add_element(shape=(), dtype=tf.int32, name='existance')

    def add_element(self, shape, dtype, name='existance'):
      self.tensor_info.append({
        'shape': shape,
        'dtype': dtype,
        'name': name,
      })

    def zeros(self):
      return [np.zeros(e['shape']) for e in self.tensor_info]

    def transform(self, obj):
      pass


  def __init__(self,
               vector_size,
               unit_model,
               ):
    super(SOCUnit, self).__init__()
    self.vector_size = vector_size
    self.unit_model = unit_model
    self.dropout_var = None

  def __call__(self, input_tensor, state):
    model_o = self.model(input_tensor, self.dropout_var)
    return model_o, state

  def model(self, input_tensor, dropout_var):
    return self.unit_model.build(self, input_tensor, dropout_var)

  def transform(self, obj):
    if obj is None:
      return self.tensor_shape.zeros()
    else:
      return self.tensor_shape.transform(obj)

  def transform(self, obj):
    return ([int(obj or 0)],)

  @property
  def state_size(self):
      return self.vector_size

  @property
  def output_size(self):
      return self.vector_size


####
class SOCListUnit(SOCUnit):

  class ListUnitShape(UnitShape):
    def __init__(self, max_length, elem, optional):
      super().__init__(optional)
      self.add_element(shape=(), dtype=tf.int32, name='seq_len')
      for ei in self.tensor_info:
        self.add_element(shape=(max_length,)+ei['shape'],
                         dtype=ei['dtype'],
                         name='list[%d].%s'%(max_length, ei['name']))

    def transform(self, obj):
      output = super().transform(obj_list)
      output.append((len(obj_list),))
      output += [np.stack(al) for al in zip(*[elem.transform(o) for o in obj])]
      return output

  def __init__(self,
               vector_size,
               unit_model,
               list_length,
               elem,
               ):
    super().__init__(vector_size, unit_model)
    self.elem = elem
    self.tensor_shape = [((list_length,) + s, t) for s, t in self.elem.tensor_shape] + [((), tf.int32), ((), tf.int32)]
    self.list_length = list_length

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
class SOCDictUnit(SOCUnit):
  def __init__(self,
               vector_size,
               unit_model,
               struct,
               ):
    super(SOCDictUnit, self).__init__(vector_size, unit_model)
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

  def transform(self, obj):
    obj = obj or {}
    res = [None] * len(self.tensor_shape)
    for key in self.key_list:
      sub_input = self.struct[key].transform(obj.get(key, None))
      for i, t in zip(self.tensor_map[key], sub_input):
        res[i] = t
    return res


####
class SOCIntegerUnit(SOCUnit):
  def __init__(self,
               vector_size,
               unit_model,
               ):
    super(SOCIntegerUnit, self).__init__(vector_size, unit_model)
    self.tensor_shape = [((1,), tf.int32)]


####
class SOCFloatUnit(SOCUnit):
  def __init__(self,
               vector_size,
               unit_model,
               ):
    super(SOCFloatUnit, self).__init__(vector_size, unit_model)
    self.tensor_shape = [((1,), tf.float32)]


####
class SOCStringUnit(SOCUnit):
  def __init__(self,
               vector_size,
               unit_model,
               string_length,
               string_encoder,
               ):
    super(SOCStringUnit, self).__init__(vector_size, unit_model)
    self.string_encoder = string_encoder
    self.string_length = string_length
    self.string_encoder = string_encoder
    self.tensor_shape = [((string_length, ), tf.int32)]

  def transform(self, obj):
    obj = obj or ''
    return [self.string_encoder(obj[-self.string_length:], self.string_length)]


####
class SOCImageUnit(SOCUnit):
  def __init__(self,
               vector_size,
               unit_model,
               base_dir,
               ):
    super(SOCImageUnit, self).__init__(vector_size, unit_model)
    self.tensor_shape = [((unit_model.bottleneck_size, ), tf.float32)]
    self.base_dir = base_dir

  def transform(self, obj):
    if obj is None:
      return ([0.0]*2048,)
    return (obj,)
