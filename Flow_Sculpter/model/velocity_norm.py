
import tensorflow as tf

def calc_velocity_norm(velocity_field):
  if len(velocity_field.get_shape()) == 4:
    vel_norm = tf.sqrt(tf.square(velocity_field[:,:,:,0]) + tf.square(velocity_field[:,:,:,1]))
  elif len(velocity_field.get_shape()) == 5:
    vel_norm = tf.sqrt(tf.square(velocity_field[...,0]) + tf.square(velocity_field[...,1]) + tf.square(velocity_field[...,2]))
  return vel_norm



