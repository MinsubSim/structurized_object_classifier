import tensorflow as tf
from . import UnitModule



class InceptionV3Module(UnitModule):
  def __init__(self, model_path):
    self.input_width = 299
    self.input_height = 299
    self.input_depth = 3
    self.input_mean = 128
    self.input_std = 128
    self.bottleneck_size = 2048

    self.input_data = tf.placeholder(tf.string, name='ImageInput')
    decoded_image = tf.image.decode_image(self.input_data, channels=self.input_depth)
    self.decoded_data = tf.cast(decoded_image, dtype=tf.float32)
    self.decoded_input = tf.placeholder(tf.float32,
                                        shape=(None, None, None),
                                        name='DecodedImageInput')
    decoded_4d = tf.expand_dims(self.decoded_input, 0)
    resize_shape = tf.stack([self.input_height, self.input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_4d, resize_shape_as_int)
    offset_image = tf.subtract(resized_image, self.input_mean)
    self.resized_data = tf.multiply(offset_image, 1.0 / self.input_std)
    with tf.gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      self.bottleneck_tensor, self.resized_input_tensor = tf.import_graph_def(
          graph_def, name='', return_elements=['pool_3/_reshape:0','Mul:0'])

  def decode(self, filename):
    if filename:
      with tf.Session() as sess:
        raw_data = tf.gfile.FastGFile(filename, 'rb').read()
        decoded_data = sess.run(self.decoded_data, feed_dict={self.input_data: raw_data})
        if filename.endswith('gif'):
          decoded_data = decoded_data[-1]
        resized_data = sess.run(self.resized_data, feed_dict={self.decoded_input: decoded_data})
        bottleneck_values = sess.run(self.bottleneck_tensor, {self.resized_input_tensor: resized_data})
        return np.squeeze(bottleneck_values, axis=0)
    else:
      return np.zeros(self.bottleneck_size)

  def build(self, unit, input_tensor, dropout_var, optional=True):
    if optional:
      existance = input_tensor[0]
      image_tensor = input_tensor[1]
      image_tensor = tf.concat([tf.cast(existance, tf.float32), image_tensor], axis=1)
    else:
      image_tensor = input_tensor[0]

    with tf.name_scope("InceptionV3Module"):
      W = tf.get_variable('W',
                          shape=(self.bottleneck_size + 1, unit.vector_size),
                          initializer=tf.glorot_normal_initializer())
      b = tf.get_variable('b',
                          shape=(unit.vector_size,),
                          initializer=tf.glorot_normal_initializer())
      result = tf.matmul(image_tensor, W) + b
      return result
