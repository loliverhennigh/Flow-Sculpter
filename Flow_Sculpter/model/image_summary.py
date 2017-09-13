
import tensorflow as tf

def image_summary(name, image):
  if len(image.get_shape()) == 5:
    image = image[:,:,:,image.get_shape()[3]/2]
  with tf.device('/cpu:0'):
    tf.summary.image(name, image)

