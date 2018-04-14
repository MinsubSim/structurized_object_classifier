import tensorflow as tf
from . import SOCFeature



class SOCNumberFeature(SOCFeature):
    ''' abstract number feature '''


class SOCIntegerFeature(SOCNumberFeature):
    def __init__(self, module=None, vector_depth=None, optional=False):
        super().__init__(module=module, optional=optional)
        self.vector_depth = vector_depth
        self.add_element(shape=[1], dtype=tf.int32, name='integer')


class SOCFloatFeature(SOCNumberFeature):
    def __init__(self, module=None, vector_depth=None, optional=False):
        super().__init__(module=module, optional=optional)
        self.vector_depth = vector_depth
        self.add_element(shape=[1], dtype=tf.float32, name='float')
