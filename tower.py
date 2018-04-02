import tensorflow as tf
import numpy as np



class SOCTower(object):
    def __init__(self, struct):
        self.struct = struct
        self.input_tensor = [tf.placeholder(shape=[None] + s['shape'], dtype=s['dtype']) for s in self.struct.tensor_shape]
        self.initial_state = tf.placeholder(shape=[None, self.struct.module.num_units], dtype=tf.float32)
        self.output = struct(self.input_tensor, self.initial_state)
    
    def feed(self, input_data, training=True):
        feed_dict = {}
        input_transformed = self.struct.transform(input_data)
        for tensor, data in zip(self.input_tensor, input_transformed):
            feed_dict[tensor] = np.asarray(data, tensor.dtype.as_numpy_dtype)
        feed_dict.update(self.struct.dropout(training))
        return self.input_tensor, feed_dict