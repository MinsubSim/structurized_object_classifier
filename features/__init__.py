import tensorflow as tf
import numpy as np
from ..tower import SOCTower



class SOCFeature(tf.contrib.rnn.RNNCell):
    ''' abstract number feature '''
    def __init__(self, module, optional=False):
        super().__init__()
        self.module = module
        self.dropout_var = tf.placeholder(dtype=tf.float32)
        self.tensor_shape = []
        self.optional = optional
        self.input_tensor = None
        if optional:
            self.add_element(shape=[1], dtype=tf.int32, name='existance')

    def __call__(self, input_tensor, state):
        with tf.variable_scope(name_or_scope=None, default_name=type(self).__name__):
            model_output, new_state = self.module.build(self, input_tensor, state)
        return model_output, new_state

    # tensor shape
    def add_element(self, shape, dtype, name):
        self.tensor_shape.append({
            'shape': shape,
            'dtype': dtype,
            'name': name,
        })

    def build_tower(self, batch_size):
        return SOCTower(self, batch_size)
    
    def dropout(self, training=True): #TODO: recursive하게 다른애들도 구현
        feed_dict = {}
        if self.module:
            if training:
                feed_dict[self.dropout_var] = self.module.dropout_val
            else:
                feed_dict[self.dropout_var] = 1.0
        return feed_dict

    def transform(self, input_data):
        return [[input_data]]

    def zeros(self):
        return [[0]]

    @property
    def state_size(self):
        return self.module.num_units

    @property
    def output_size(self):
        return self.module.num_units



from . import number, string, sequence, hashmap
