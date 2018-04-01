import tensorflow as tf
from . import SOCFeature



class SOCNumberFeature(SOCFeature):
    ''' abstract number feature '''


class SOCIntegerFeature(SOCNumberFeature):
    ''' abstract number feature '''
    def __init__(self, module, vector_size, optional=False):
        super().__init__(module=module, vector_size=vector_size, optional=optional)
        self.shape_add_element(shape=(1,), dtype=np.int32, name='integer')


class SOCFloatFeature(SOCNumberFeature):
    ''' abstract number feature '''
    def __init__(self, module, vector_size, optional=False):
        super().__init__(module=module, vector_size=vector_size, optional=optional)
        self.shape_add_element(shape=(1,), dtype=np.float32, name='float')
