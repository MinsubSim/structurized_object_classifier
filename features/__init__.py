import tensorflow as tf
import numpy as np



class SOCFeature(tf.contrib.rnn.RNNCell):
    ''' abstract number feature '''
    def __init__(self, module, vector_size, optional=False):
        super(SOCFeature, self).__init__()
        self.vector_size = vector_size
        self.module = module
        self.dropout_var = None
        self.tensor_shape = []
        self.optional = optional
        if optional:
            self.shape_add_element(shape=(1,), dtype=np.int32, name='existance')

    def __call__(self, input_tensor, state):
        model_o = self.model(input_tensor, self.dropout_var)
        return model_o, state

    # create model
    def model(self, input_tensor, dropout_var):
        return self.module.build(self, input_tensor, dropout_var)

    # tensor shape
    def shape_add_element(self, shape, dtype, name):
        self.tensor_shape.append({
            'shape': shape,
            'dtype': dtype,
            'name': name,
        })

    # create empty input for this Feature
    def zeros(self):
        return [np.zeros(shape=e['shape'], dtype=e['dtype']) for e in self.tensor_shape]

    # create input of object for this Feature
    def transform(self, obj):
        output = []
        if obj is None:
            if self.optional:
                output += self.zeros()
            else:
                raise ValueError('non-optional attribute is missing in %s' % (str(self)))
        else:
            if self.optional:
                output.append(np.asarray([1]))
            output += self._transform(obj)
        return output

    def _transform(self, obj):
        return [np.asarray([obj])]

    @property
    def state_size(self):
            return self.vector_size

    @property
    def output_size(self):
            return self.vector_size

#TODO: transform 코드 전반적으로 훑어보면서 pandas로 처리할거있는지 확인
