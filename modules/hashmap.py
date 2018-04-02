import tensorflow as tf
from . import UnitModule



class ConcatFCNModule(UnitModule):
    def build(self, feature, input_tensor, state):
        all_output = []
        for k in feature.key_list:
            elem_input = [input_tensor[i] for i in feature.elem_index(k)]
            output, state = feature.struct[k](elem_input, state)
            all_output.append(output)
        fcl_res = tf.contrib.layers.fully_connected(inputs=tf.concat(all_output, axis=1),
                                                    num_outputs=self.num_units)
        return tf.contrib.layers.dropout(inputs=fcl_res, keep_prob=feature.dropout_var), state
