import json, random
data_dir = '/root/kastera/kastera/report_load'

objs = []

label_name = ['NORMAL','INSULT+SEXUAL','AD','PORN','GAMBLE','ETC','SEXUAL','PLASTER']
selected_labels =  [0, 1, 2, 3, 4, 7]

with open(data_dir+'/data_th/openlink/sum', 'r') as f:
    cnt = 0
    while True:
        line = f.readline()
        if not line:
            break

        jdat = json.loads(line)
        label = jdat.get('spam_type', 0)

        if label not in selected_labels:
            continue

        if not jdat['_chat_log_info']:
            continue
        objs.append(jdat)

print(len(objs))
print(objs[0])

h = {}
for o in objs:
    key = (o['participant_id'], o['link_id'])
    if key in h:
        h[key].append(o)
    else:
        h[key] = [o]

bad_h = {}
good_l = []
for k,v in h.items():
    if len(v) > 1:
        ss = set(x['spam_type'] for x in v)
        if len(ss) > 1:
            bad_h[k] = v
            continue
    good_l += v


cat_h = {}
for l in range(8):
    cat_h[l] = []

for jdat in good_l:
    ndat = []
    p_cnt = 0
    for c in jdat['_chat_log_info']:
        if c['u'] == jdat['participant_id']:
            p_cnt += 1
        if 'm' in c:
            
            if type(c['m']) is str:
                t = {
                    'm': str(c['m']),
                    's': int(c['s']),
                    'u': int(c['u']==jdat['participant_id']),
                }
                ndat.append(t)
            elif type(c['m']) is dict:
                t={
                    's': int(c['s']),
                    'u': int(c['u']==jdat['participant_id']),
                }
                if 'message' in c['m']:
                    t['m'] = str(c['m']['message'])
                if 'thumbnailUrl::img' in c['m']:
                    t['img'] = str(c['m']['thumbnailUrl::img'])
                elif 'attachment' in c['m'] and 'thumbnailUrl::img' in c['m']['attachment']:
                    t['img'] = str( c['m']['attachment']['thumbnailUrl::img'])
                ndat.append(t)
            
    #ndat['chat_log'] = ' '.join(ndat['chat_log']).replace('\n', ' ')
    if len(ndat) == 0 or p_cnt == 0:
        #print(jdat)
        continue
        
    ndat = sorted(ndat, key=lambda x: x['s'])
    s = ndat[0]['s']
    for x in ndat:
        ts = x['s']
        x['s'] = ts - s
        s = ts
    label = jdat.get('spam_type', 0)

    ndat2 = {
        'data': ndat,
        'meta': [jdat['participant_id'], jdat['id']]
    }
    cat_h[label].append(ndat2)

cat_h[1] += cat_h[6]
cat_h.pop(6)

from units import *
import modules
from model import *
import pickle

base_dir='/root/kastera/kastera/report_load/'
img_model = modules.image.InceptionV3Module(model_path='classify_image_graph_def.pb')

print('start!')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    for label, lobjs in cat_h.items():
        for obj in lobjs:
            o = obj['data']
            for x in o:
                if 'img' in x:
                    data = img_model.decode(base_dir+x['img'], sess)
                    x['img'] = data
                else:
                    data = img_model.decode(None, sess)
                    x['img'] = data

objs = []
cc = 0
for l in selected_labels:
    print(l, len(cat_h[l]), len(objs))
    random.shuffle(cat_h[l])
    t = []
    while len(t) < 10000:
        t += [(x['data'], cc, x['meta']) for x in cat_h[l]]
    objs += t[:10000]
    cc += 1
print(len(objs))
random.shuffle(objs)

    #img_model.decode()

with open('soc_Objs', 'wb') as f:
    pickle.dump(objs, f)
print('insert finish')


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
      'img': SOCImageUnit(vector_size=100,
                          unit_model=modules.image.InceptionV3Module(model_path='classify_image_graph_def.pb'),
                          base_dir='/root/kastera/kastera/report_load/',
                         )
    })
    )
soc_model = SOCModel(struct=obj_structure, dropout_prob=0.9, label_size=len(selected_labels), learning_rate=1e-4, num_gpus=2)
input_data, label_data, meta_data = zip(*objs)
soc_model.insert(input_data, label_data, meta_data)
alldata = soc_model.data_stack
with open('soc_Data', 'wb') as f:
    pickle.dump(alldata, f)
print('insert finish')
