import tensorflow as tf
from . import SOCFeature



class SOCDictFeature(SOCFeature):
    ''' abstract number feature '''
    def __init__(self, module, struct, optional=False):
        super().__init__(module=module, optional=optional)
        key_list = sorted(struct.keys())
        tensor_index_map = {}
        tensor_index = 0
        for k in key_list:
            tensor_index_map[k] = []
            for shape in struct[k].tensor_shape:
                tensor_index_map[k].append(tensor_index)
                tensor_index += 1
                self.add_element(
                    shape=shape['shape'],
                    dtype=shape['dtype'],
                    name='dict{%s}.%s' % (k, shape['name'])
                    )
        self.struct = struct
        self.key_list = key_list
        self.tensor_index_map = tensor_index_map
        self.tensor_size = tensor_index

    def transform(self, input_data):
        input_data = input_data or {}
        res = [None] * self.tensor_size
        for key in self.key_list:
            val = input_data.get(key, None)
            if val is None: #TODO: pandas nan 도 체크
                sub_input = self.struct[key].zeros()
            else:
                sub_input = self.struct[key].transform()
            for i, t in zip(self.tensor_index_map[key], sub_input):
                res[i] = t
        if any(r is None for r in res):
            raise ValueError('transform does not return complete tensor')
        return res

    def elem_index(self, sub_key):
        return self.tensor_index_map[sub_key]
