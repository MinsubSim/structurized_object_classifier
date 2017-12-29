from cell import *
from model import *

obj_structure = SOCListCell(
    vector_size=100,
    list_length=200,
    elem=SOCDictCell(
      vector_size=100,
      struct={
        'is_suspect': SOCIndexCell(vector_size=100),
        'create_at': SOCFigureCell(vector_size=100),
        'msg': SOCStringCell(vector_size=100,
                             string_length=100,
                             embedding_size=300,
                             char_depth=12345,
                             filter_size=30),
        #'img': ImageStruct()
      })
      )

import random
soc_model = SOCModel(struct=obj_structure, dropout_prob=0.9, label_size=2, learning_rate=1e-4)

data_list = []

for i in range(500):
  obj = []
  cur_time = random.randint(100000,200000)
  for j in range(30):
    cur_time += random.randint(1000,10000)
    msg = ''.join(chr(random.randint(0,25)+ord('a')) for _ in range(random.randint(50, 100)))
    t = {
      'is_suspect': random.randint(0,1),
      'create_at': cur_time,
      'msg': msg,
    }
    obj.append(t)
  data_list.append((obj, 0))

for i in range(500):
  obj = []
  cur_time = random.randint(100000,200000)
  msg = ''.join(chr(random.randint(0,25)+ord('a')) for _ in range(random.randint(1, 30)))
  for j in range(30):
    cur_time += random.randint(10,100)
    iss = 0 if random.randint(0,10) == 0 else 1
    msg = ''.join(chr(random.randint(0,25)+ord('a')) for _ in range(random.randint(50, 100)))
    t = {
      'is_suspect': 0 if random.randint(0,10) == 0 else 1,
      'create_at': cur_time,
      'msg': ''.join(chr(random.randint(0,25)+ord('a')) for _ in range(random.randint(50, 100))),
    }
    if iss == 1:
      t['msg'] = msg * random.randint(1, 10)
    obj.append(t)
  data_list.append((obj, 1))

random.shuffle(data_list)

input_data, label_data = zip(*data_list)

#print(input_data[:10])
#print(label_data[:10])

soc_model.insert(input_data, label_data)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.allow_soft_placement = True


with tf.Session(config=config) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)

  for _ in range(100):
    input_data, label_data = soc_model.batch(10)
    feed_dict = {
      soc_model.label_tensor: label_data,
      soc_model.dropout_var: 0.9,
    }

    for x, y in zip(soc_model.input_tensor, input_data):
      feed_dict[x] = y

    res = sess.run(soc_model.eval_set, feed_dict=feed_dict)
    print(res)
