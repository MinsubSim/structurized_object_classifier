
class ConcatFCNModel(UnitModel):
  def struct(self, unit, input_tensor, dropout_var):
    with tf.name_scope("DictStruct"):
      all_o = []
      for k in unit.key_list:
        sub_input = [input_tensor[i] for i in unit.tensor_map[k]]
        out = unit.struct[k].model(sub_input, dropout_var)
        all_o.append(out)
      merged_i = tf.concat(all_o, axis=1)
      merged_o = tf.contrib.layers.fully_connected(inputs=merged_i,
                                                   num_outputs=vector_size)
      return merged_o
