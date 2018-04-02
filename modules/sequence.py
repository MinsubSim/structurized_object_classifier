import tensorflow as tf
from . import UnitModule



class BasicRNNModule(UnitModule):
    def __init__(self, num_cells, attention=[]):
        self.num_cells = num_cells
        self.attention = attention

    def build(self, feature, input_tensor, state):
        cells = [feature.elem]
        for i in range(self.num_cells):
            cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units)
            if i in self.attention:
                cell = tf.contrib.rnn.AttentionCellWrapper(cell, 32, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=feature.dropout_var)
            cells.append(cell)
        elem_cell = tf.contrib.rnn.MultiRNNCell(cells)

        sequence_length = input_tensor[0]
        data_tensor = input_tensor[1:]
        output, new_state = tf.nn.dynamic_rnn(cell=elem_cell,
                                              inputs=data_tensor,
                                              initial_state=state,
                                              sequence_length=sequence_length,
                                              dtype=tf.float32)

        batch_range = tf.range(tf.shape(output)[0])
        indices = tf.stack([batch_range, sequence_length-1], axis=1)
        res = tf.gather_nd(output, indices)
        return res, new_state

#TODO: dropout variable을 어떻게 처리할 것인가
#TODO: 현재는 shape만 다 저장해놓고 맨마지막에 다 만들어서 집어넣음
#TODO: dropout variable 은 elem별로 각각 만들어놓고, input shape으로 텐서 만들어서 feed dict채우면 거기다가 추가로 채워주는 메소드 만들면 될듯
#TODO: feed dict채워주는 메소드 구현해서 그안에 넣자
#TODO: pandas dataframe 지원가능하도록 구현


class OneHotReduceModule(UnitModule):
    def build(self, feature, input_tensor, state):
        data_tensor = input_tensor[1:]
        one_hot = tf.one_hot(data_tensor, feature.vector_depth, on_value=True, off_value=False)
        reduce_out = tf.cast(tf.reduce_any(one_hot, axis=0), tf.float32)
        fcl_res = tf.contrib.layers.fully_connected(inputs=tf.concat(reduce_out, axis=1),
                                                    num_outputs=self.num_units)
        return tf.contrib.layers.dropout(inputs=fcl_res, keep_prob=feature.dropout_var), state
