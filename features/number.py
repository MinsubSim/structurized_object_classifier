import tensorflow as tf
import numpy as np
from . import SOCFeature



class SOCNumberFeature(SOCFeature):
    ''' abstract number feature '''


class SOCIntegerFeature(SOCNumberFeature):
    ''' abstract number feature '''
    def __init__(self, module, optional=False):
        super().__init__(module=module, optional=optional)
        self.add_element(shape=[1], dtype=np.int32, name='integer')


class SOCFloatFeature(SOCNumberFeature):
    ''' abstract number feature '''
    def __init__(self, module, optional=False):
        super().__init__(module=module, optional=optional)
        self.add_element(shape=[1], dtype=np.float32, name='float')
