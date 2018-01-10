import tensorflow as tf
import numpy as np



class SOCUnit(tf.contrib.rnn.RNNCell):
  def __init__(self,
               unit_model,
               vector_size,
               optional=False,
               ):
    super(SOCUnit, self).__init__()
    self.vector_size = vector_size
    self.unit_model = unit_model
    self.dropout_var = None
    self.tensor_shape = []
    self.optional = optional
    if optional:
      self.shape_add_element(shape=(), dtype=np.int32, name='existance')

  def __call__(self, input_tensor, state):
    model_o = self.model(input_tensor, self.dropout_var)
    return model_o, state

  # create model
  def model(self, input_tensor, dropout_var):
    return self.unit_model.build(self, input_tensor, dropout_var)

  # tensor shape
  def shape_add_element(self, shape, dtype, name):
    self.tensor_shape.append({
      'shape': shape,
      'dtype': dtype,
      'name': name,
    })

  # create empty input for this unit
  def zeros(self):
    return [np.zeros(shape=e['shape'], dtype=e['dtype']) for e in self.tensor_shape]

  # create input of object for this unit
  def transform(self, obj):
    output = []
    if obj is None:
      if self.optional:
        output += self.zeros()
      else:
        raise 'non-optional attribute is missing'
    else:
      if self.optional:
        output.append(np.asarray(1))
      output += self._transform(obj)
    return output

  def _transform(self, obj):
    return [np.asarray(obj)]

  @property
  def state_size(self):
      return self.vector_size

  @property
  def output_size(self):
      return self.vector_size



####
class SOCListUnit(SOCUnit):
  def __init__(self,
               unit_model,
               vector_size,
               list_length,
               elem,
               optional=False,
               ):
    super().__init__(unit_model=unit_model,
                     vector_size=vector_size,
                     optional=optional)
    self.elem = elem
    self.list_length = list_length

    self.shape_add_element(shape=(), dtype=np.int32, name='seq_len')
    for shape in self.elem.tensor_shape:
      self.shape_add_element(shape=(list_length,)+shape['shape'],
                             dtype=shape['dtype'],
                             name='list[%d].%s'%(list_length, shape['name']))

  def _transform(self, obj):
    output = []
    output.append(np.asarray(len(obj)))
    t = [self.elem.transform(o) for o in obj[-self.list_length:]]
    t += [self.elem.zeros()] * (self.list_length - len(obj))
    output += [np.stack(al) for al in zip(*t)]
    return output


####
class SOCDictUnit(SOCUnit):
  def __init__(self,
               unit_model,
               vector_size,
               struct,
               optional=False,
               ):
    super().__init__(unit_model=unit_model,
                     vector_size=vector_size,
                     optional=optional)
    key_list = sorted(struct.keys())
    tensor_index_map = {}
    tensor_index = 0
    for k in key_list:
      tensor_index_map[k] = []
      for shape in struct[k].tensor_shape:
        tensor_index_map[k].append(tensor_index)
        tensor_index += 1
        self.shape_add_element(shape=shape['shape'],
                               dtype=shape['dtype'],
                               name='dict{%s}.%s'%(k, shape['name']))
    self.struct = struct
    self.key_list = key_list
    self.tensor_index_map = tensor_index_map
    self.tensor_size = tensor_index

  def _transform(self, obj):
    obj = obj or {}
    res = [None] * self.tensor_size
    for key in self.key_list:
      sub_input = self.struct[key].transform(obj.get(key, None))
      for i, t in zip(self.tensor_index_map[key], sub_input):
        res[i] = t
    if any(r is None for r in res):
      raise 'transform does not return complete tensor'
    return res


####
class SOCIntegerUnit(SOCUnit):
  def __init__(self,
               unit_model,
               vector_size,
               optional=False,
               ):
    super().__init__(unit_model=unit_model,
                     vector_size=vector_size,
                     optional=optional)
    self.shape_add_element(shape=(),
                           dtype=np.int32,
                           name='integer')


####
class SOCFloatUnit(SOCUnit):
  def __init__(self,
               unit_model,
               vector_size,
               optional=False,
               ):
    super().__init__(unit_model=unit_model,
                     vector_size=vector_size,
                     optional=optional)
    self.shape_add_element(shape=(),
                           dtype=np.float32,
                           name='float')


####
class SOCStringUnit(SOCUnit):
  def __init__(self,
               unit_model,
               vector_size,
               string_length,
               string_encoder,
               optional=False,
               ):
    super().__init__(unit_model=unit_model,
                     vector_size=vector_size,
                     optional=optional)
    self.string_encoder = string_encoder
    self.string_length = string_length
    self.string_encoder = string_encoder
    self.shape_add_element(shape=(string_length,),
                           dtype=np.int32,
                           name='string')

  def _transform(self, obj):
    return [np.asarray(self.string_encoder(obj[-self.string_length:], self.string_length))]


####
class SOCImageUnit(SOCUnit):
  def __init__(self,
               unit_model,
               vector_size,
               base_dir,
               optional=False,
               ):
    super().__init__(unit_model=unit_model,
                     vector_size=vector_size,
                     optional=optional)
    self.base_dir = base_dir
    self.shape_add_element(shape=(unit_model.bottleneck_size,),
                           dtype=np.float32,
                           name='image')
