import tensorflow as tf
from . import SOCFeature


####
class SOCSequenceFeature(SOCFeature):
  ''' abstract number feature '''
  def __init__(self,
               module,
               vector_size,
               list_length,
               elem,
               optional=False,
               ):
    super().__init__(module=module,
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
