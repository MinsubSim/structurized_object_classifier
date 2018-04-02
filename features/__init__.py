import tensorflow as tf
import numpy as np
from ..tower import SOCTower



class SOCFeature(tf.contrib.rnn.RNNCell):
    ''' abstract number feature '''
    def __init__(self, module, optional=False):
        super(SOCFeature, self).__init__()
        self.module = module
        self.dropout_var = tf.placeholder(dtype=tf.float32)
        self.tensor_shape = []
        self.optional = optional
        self.input_tensor = None
        if optional:
            self.add_element(shape=[1], dtype=np.int32, name='existance')


    def __call__(self, input_tensor, state):
        model_output, new_state = self.module.build(self, input_tensor, state)
        return model_output, new_state

    # tensor shape
    def add_element(self, shape, dtype, name):
        self.tensor_shape.append({
            'shape': shape,
            'dtype': dtype,
            'name': name,
        })

    def build_tower(self):
        return SOCTower(self)
    
    def dropout(self, training=True): #TODO: recursive하게 다른애들도 구현
        feed_dict = {}
        if training:
            feed_dict[self.dropout_var] = self.module.dropout_val
        else:
            feed_dict[self.dropout_var] = 1.0
        return feed_dict

    def transform(self, input_data): #TODO: pandas로 구현
        pass

    def zeros(self):
        pass

    @property
    def state_size(self):
        return self.vector_size

    @property
    def output_size(self):
        return self.vector_size



from . import number, string, sequence, hashmap