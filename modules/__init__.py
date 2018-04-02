import tensorflow as tf



class UnitModule:
    def __init__(self, num_units, dropout_val=0.9):
        self.num_units = num_units
        self.dropout_val = dropout_val

    def build(self, feature, input_tensor, state):
        pass

from . import number, string, sequence, hashmap