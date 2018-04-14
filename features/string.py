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
        self.add_element(shape=[1], dtype=tf.int32, name='str_len')
        self.add_element(shape=[string_length],
                         dtype=tf.int32,
                         name='string')

    def transform(self, obj):
        output = [[len(obj)]]
        if self.string_vectorizer is None:
            output.append(obj + [self.vector_depth] * (self.string_length - len(obj)))
        else:
            t = obj.map(lambda x: self.string_vectorizer(x[-self.string_length:], self.string_length))
            t += [self.vector_depth] * (self.string_length - len(t))
        return output

    
    def zeros(self):
        return [[0], [self.vector_depth] * self.string_length]
