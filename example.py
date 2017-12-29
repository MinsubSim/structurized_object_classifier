from structure import *

obj_structure = SOCDictCell(
    struct={
      'is_suspect': SOCIndexCell(vector_size=100),
      'create_at': SOCFigureCell(vector_size=100),
      'msg': SOCStringCell(vector_size=100,
                           string_length=2000,
                           embedding_size=300,
                           char_depth=12345,
                           filter_size=123),
      #'img': ImageStruct()
    },
    vector_size=100,
  )

soc_model = SOCModel(struct=obj_structure, dropout_prob=0.9, label_size=2)

soc_model.feed(input_data, label_data)
soc_model.train(10)
