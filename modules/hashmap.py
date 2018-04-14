import tensorflow as tf
from . import UnitModule



class ConcatFCNModule(UnitModule):
    def build(self, feature, input_tensor, state):
        all_output = []
        for k in feature.key_list:
            elem_input = [input_tensor[i] for i in feature.elem_index(k)]
            output, state = feature.struct[k](elem_input, state)
            all_output.append(output)
        output = tf.concat(all_output, axis=1)
        output = tf.contrib.layers.fully_connected(inputs=output, num_outputs=self.num_units)
        output = tf.contrib.layers.fully_connected(inputs=output, num_outputs=self.num_units)
        return tf.contrib.layers.dropout(inputs=output, keep_prob=feature.dropout_var), state
