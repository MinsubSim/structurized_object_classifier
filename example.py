from units import *
from model import *
import modules
import pickle
import random

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--list_vector', type=int)
parser.add_argument('--list_length', type=int)
parser.add_argument('--dict_vector', type=int)
parser.add_argument('--string_vector', type=int)
parser.add_argument('--string_embedding', type=int)
parser.add_argument('--string_filter', type=int)
parser.add_argument('--string_length', type=int)
parser.add_argument('--image_vector', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--num_gpu', type=int)
parser.add_argument('--drop_out', type=float)
parser.add_argument('--loss_func', type=str)

args = parser.parse_args()
print(args)

selected_labels =  [0, 1, 2, 3, 4, 7]

with open('soc_Objs2', 'rb') as f:
    objs = pickle.load(f)
print('insert finish')
random.shuffle(objs)

tf.reset_default_graph()
obj_structure = SOCListUnit(
  vector_size=args.list_vector,
  unit_model=modules.sequence.BasicRNNModule(),
  list_length=args.list_length,
  elem=SOCDictUnit(
    vector_size=args.dict_vector,
    unit_model=modules.hashmap.ConcatFCNModule(),
    struct={
      'u': SOCIntegerUnit(vector_size=2,
                          unit_model=modules.number.OneHotModule()),
      's': SOCIntegerUnit(vector_size=5,
                          unit_model=modules.number.ExpansionModule()),
      'm': SOCStringUnit(vector_size=args.string_vector,
                         unit_model=modules.string.BasicCNNModule(char_depth=11682,
                                                                  embedding_size=args.string_embedding,
                                                                  filter_size=args.string_filter),
                         string_encoder=modules.string.string_encode,
                         string_length=args.string_length,
                         optional=True,
                         ),
      'img': SOCImageUnit(vector_size=args.image_vector,
                          unit_model=modules.image.InceptionV3Module(model_path='classify_image_graph_def.pb'),
                          base_dir='files',
                          optional=True,
                          )
    })
    )

soc_model = SOCModel(struct=obj_structure, dropout_prob=args.drop_out, label_size=len(selected_labels), learning_rate=args.learning_rate, num_gpus=args.num_gpu, loss_func=args.loss_func)

print('start!', flush=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

saver = tf.train.Saver(tf.global_variables())

with tf.Session(config=config) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)

  for idx in range(10):
    tobjs = objs[:5000]
    del objs[:5000]
    input_data, label_data, meta_data = zip(*tobjs)
    soc_model.insert(input_data, label_data, meta_data)
    for i in range(500):
      tt = soc_model.train(sess, 5)
      _, loss, acc, pred = tt[0]
      print(idx, i, acc, flush=True)
      print(loss)
      for x in zip(tt[1], pred, tt[2]):
          print(x, flush=True)
    saver.save(sess, 'my-model', global_step=idx)


