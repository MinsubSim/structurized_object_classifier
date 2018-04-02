import tensorflow as tf
import numpy as np
from . import SOCFeature



class SOCStringFeature(SOCFeature):
    ''' abstract number feature '''
    def __init__(self,
                 module,
                 vector_depth,
                 string_length,
                 string_vectorizer=None,
                 optional=False):
        super().__init__(module=module, optional=optional)
        self.vector_depth = vector_depth
        self.string_length = string_length
        self.string_vectorizer = string_vectorizer
        self.add_element(shape=[string_length],
                         dtype=np.int32,
                         name='string')

    def _transform(self, obj):
        if self.string_vectorizer is None:
            return [np.asarray(obj)]
        return [np.asarray(self.string_encoder(obj[-self.string_length:], self.string_length))]