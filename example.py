from structure import *

obj_structure =
  DictStruct(
    shape={
      'aaa': ListStruct(
        elem=DictStruct(
          shape={
            'a': StringStruct(vector_size=100, embedding_size=300),
            'b': FigureStruct(vector_size=100),
            'c': IndexStruct(vector_size=100),
          },
          vector_size=100,
        ),
        vector_size=100,
      )
    },
    vector_size=100,
  )
