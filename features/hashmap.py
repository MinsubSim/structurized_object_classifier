import tensorflow as tf
from . import SOCFeature



class SOCHashmapFeature(SOCFeature):
    ''' abstract number feature '''
    def __init__(self, module, vector_size, struct, optional=False):
        super().__init__(module=module, vector_size=vector_size, optional=optional)
        key_list = sorted(struct.keys())
        tensor_index_map = {}
        tensor_index = 0
        for k in key_list:
            tensor_index_map[k] = []
            for shape in struct[k].tensor_shape:
                tensor_index_map[k].append(tensor_index)
                tensor_index += 1
                self.shape_add_element(
                    shape=shape['shape'],
                    dtype=shape['dtype'],
                    name='dict{%s}.%s' % (k, shape['name'])
                    )
        self.struct = struct
        self.key_list = key_list
        self.tensor_index_map = tensor_index_map
        self.tensor_size = tensor_index

    def _transform(self, obj):
        obj = obj or {}
        res = [None] * self.tensor_size
        for key in self.key_list:
            sub_input = self.struct[key].transform(obj.get(key, None))
            for i, t in zip(self.tensor_index_map[key], sub_input):
                res[i] = t
        if any(r is None for r in res):
            raise ValueError('transform does not return complete tensor')
        return res

    def get_sub_input(self, input_tensor, sub_key):
        return [input_tensor[i] for i in self.tensor_index_map[sub_key]]
