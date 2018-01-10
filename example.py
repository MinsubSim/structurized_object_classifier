import pickle
with open('soc_Sample', 'rb') as f:
  alldata = pickle.load(f)

sls =  [0, 1, 2, 4, 7]
from units import *
from model import *
import modules

tf.reset_default_graph()
obj_structure = SOCListUnit(
  vector_size=50,
  unit_model=modules.sequence.BasicRNNModule(),
  list_length=50,
  elem=SOCDictUnit(
    vector_size=50,
    unit_model=modules.hashmap.ConcatFCNModule(),
    struct={
      'u': SOCIntegerUnit(vector_size=2,
                          unit_model=modules.number.OneHotModule()),
      's': SOCIntegerUnit(vector_size=5,
                          unit_model=modules.number.ExpansionModule()),
      'm': SOCStringUnit(vector_size=50,
                         unit_model=modules.string.BasicCNNModule(char_depth=11682, embedding_size=100, filter_size=50),
                         string_encoder=modules.string.string_encode,
                         string_length=1000),
      #'img': ImageStruct()
    })
    )




soc_model = SOCModel(struct=obj_structure, dropout_prob=0.9, label_size=len(sls), learning_rate=1e-4, num_gpus=4)
soc_model.data_stack = alldata
total_data_size = len(alldata)

print('start!')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  for i in range(1000):
    print(soc_model.train(sess, 10))
    
  print('eval')
  for i in range(10):
    print(soc_model.evaluate(sess, 10))
