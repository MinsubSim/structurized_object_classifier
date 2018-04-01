import tensorflow as tf
from . import SOCFeature



class SOCStringFeature(SOCFeature):
  ''' abstract number feature '''
  def __init__(self,
               module,
               vector_size,
               string_length,
               string_encoder, #TODO: vectorizer로 수정
               optional=False,
               ):
    super().__init__(module=module,
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
