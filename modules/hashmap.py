import tensorflow as tf
from . import UnitModule



class ConcatFCNModule(UnitModule):
  def build(self, unit, input_tensor, dropout_var):
    with tf.name_scope("ConcatFCNModule"):
      all_o = []
      for k in unit.key_list:
        out = unit.struct[k].model(unit.get_sub_input(input_tensor, k), dropout_var)
        all_o.append(out)
      merged_i = tf.concat(all_o, axis=1)
      #merged_o = tf.contrib.layers.fully_connected(inputs=merged_i,
      #                                             num_outputs=unit.vector_size)
      return merged_i
