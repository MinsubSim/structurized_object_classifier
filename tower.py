import tensorflow as tf
import numpy as np



class SOCTower(object):
    def __init__(self, struct, batch_size):
        self.struct = struct
        self.input_tensor = [tf.placeholder(shape=[batch_size] + s['shape'], dtype=s['dtype'], name=s['name']) for s in self.struct.tensor_shape]
        self.initial_state = tf.get_variable('initial_state',
            shape=[batch_size, self.struct.module.num_units],
            initializer=tf.glorot_normal_initializer())
        self.output, self.state = struct(self.input_tensor, self.initial_state)
    
    def feed(self, input_data, training=True):
        feed_dict = {}
        input_transformed = []
        for d in input_data:
            data = self.struct.transform(d)
            input_transformed.append(data)

        input_transformed = zip(*input_transformed)
        for tensor, data in zip(self.input_tensor, input_transformed):
            dtype = tensor.dtype.as_numpy_dtype()
            feed_dict[tensor] = np.array(data, dtype=dtype)
        feed_dict.update(self.struct.dropout(training))
        return feed_dict
