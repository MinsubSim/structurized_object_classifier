import tensorflow as tf
import numpy as np  
from . import SOCFeature


####
class SOCSequenceFeature(SOCFeature):
    ''' abstract number feature '''
    def __init__(self,
                 module,
                 elem,
                 list_length,
                 optional=False,
                 ):
        super().__init__(module=module, optional=optional)
        self.elem = elem
        self.list_length = list_length

        self.add_element(shape=[1], dtype=tf.int32, name='seq_len')
        for shape in self.elem.tensor_shape:
            self.add_element(shape=[list_length] + shape['shape'],
                             dtype=shape['dtype'],
                             name='list_%d_%s' % (list_length, shape['name']))

    def dropout(self, training=True):
        feed_dict = super().dropout(training)
        feed_dict.update(self.elem.dropout(training))
        return feed_dict

    def transform(self, obj):
        output = [[len(obj)]]
        t = [self.elem.transform(x) for x in obj[-self.list_length:]]
        t += [self.elem.zeros()] * (self.list_length - len(obj))
        output += [al for al in zip(*t)]
        return output
    
    def zeros(self):
        output = [[0]]
        t = [self.elem.zeros()] * self.list_length
        output += [al for al in zip(*t)]
        return output
