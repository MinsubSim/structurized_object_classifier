sls =  [0, 1, 2, 4, 7]
from units import *
from model import *
import modules

tf.reset_default_graph()
obj_structure = SOCListUnit(
  vector_size=100,
  unit_model=modules.sequence.BasicRNNModule(),
  list_length=50,
  elem=SOCDictUnit(
    vector_size=100,
    unit_model=modules.hashmap.ConcatFCNModule(),
    struct={
      'u': SOCIntegerUnit(vector_size=2,
                          unit_model=modules.number.OneHotModule()),
      's': SOCIntegerUnit(vector_size=5,
                          unit_model=modules.number.ExpansionModule()),
      'm': SOCStringUnit(vector_size=100,
                         unit_model=modules.string.BasicCNNModule(char_depth=11682, embedding_size=500, filter_size=100),
                         string_encoder=modules.string.string_encode,
                         string_length=1000),
      'img': SOCImageUnit(vector_size=100,
                          unit_model=modules.image.InceptionV3Module(model_path='classify_image_graph_def.pb'),
                          base_dir='files',
                         )
    })
    )


import pickle

soc_model = SOCModel(struct=obj_structure, dropout_prob=0.9, label_size=len(sls), learning_rate=1e-4, num_gpus=2)

print('start!')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
    
  for idx in range(11):
    with open('soc_Data_%d'%(idx), 'rb') as f:
      alldata = pickle.load(f)
    soc_model.data_stack = alldata
    for i in range(250):
      tt = soc_model.train(sess, 10)
      print(idx,i,tt[0])
    
  with open('soc_Data_%d'%(11), 'rb') as f:
    alldata = pickle.load(f)
  soc_model.data_stack = alldata
  for i in range(250):
    tt = soc_model.eval(sess, 10)
    print('eval',i,tt[0])
