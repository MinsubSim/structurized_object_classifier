import numpy as np
class ImageDecoder():
    def __init__(self, model_path, input_width, input_height, input_depth, input_mean, input_std):
        self.input_data = tf.placeholder(tf.string, name='ImageInput')
        decoded_image = tf.image.decode_image(self.input_data, channels=input_depth)
        self.decoded_data = tf.cast(decoded_image, dtype=tf.float32)
        self.decoded_input = tf.placeholder(tf.float32,
                                            shape=(None, None, None),
                                            name='DecodedImageInput')
        decoded_4d = tf.expand_dims(self.decoded_input, 0)
        resize_shape = tf.stack([input_height, input_width])
        resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
        resized_image = tf.image.resize_bilinear(decoded_4d, resize_shape_as_int)
        offset_image = tf.subtract(resized_image, input_mean)
        self.resized_data = tf.multiply(offset_image, 1.0 / input_std)
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.bottleneck_tensor, self.resized_input_tensor = (tf.import_graph_def(
                  graph_def,
                  name='',
                  return_elements=['pool_3/_reshape:0','Mul:0']))

    def decode(self, filenames):
        bv_list = []
        with tf.Session() as sess:
            for filename in filenames:
                raw_data = tf.gfile.FastGFile(filename, 'rb').read()
                decoded_data = sess.run(self.decoded_data, feed_dict={self.input_data: raw_data})
                if filename.endswith('gif'):
                    decoded_data = decoded_data[-1]
                resized_data = sess.run(self.resized_data, feed_dict={self.decoded_input: decoded_data})
                bottleneck_values = sess.run(self.bottleneck_tensor, {self.resized_input_tensor: resized_data})
                bv_list.append(bottleneck_values)
        return np.concatenate(bv_list, axis=0)


model_path = '/Users/kakao/hello/seocheck/kastera/kastera/report_load/inception-2015-12-05/'
model_path = model_path + 'classify_image_graph_def.pb'
idc = ImageDecoder(model_path = model_path, input_width = 299,input_height = 299,input_depth = 3,input_mean = 128,input_std = 128)
res = idc.decode([dir_path+'/'+x for x in filenames])
