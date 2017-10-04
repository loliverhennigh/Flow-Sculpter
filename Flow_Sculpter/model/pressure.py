
import tensorflow as tf
import numpy as np

def _simple_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv2d(x, k, [1, 1, 1, 1], padding='VALID')
  return y

def _simple_conv_3d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv3d(x, k, [1, 1, 1, 1, 1], padding='SAME')
  return y

def calc_force(boundary, pressure_field):
  if len(boundary.get_shape()) == 4:
    f = calc_force_2d(boundary, pressure_field)
  else:
    f = calc_force_3d(boundary, pressure_field)
  return f

def calc_force_2d(boundary, pressure_field):
  # implementation of spatial divergence
  # reimplemented from torch FluidNet implementation

  # make weight for x divergence
  weight_np = np.zeros([3,3,1,2])
  angle_norm = 1.0 / np.sqrt(2.0)
  weight_np[0,0,0,1] =  angle_norm
  weight_np[0,1,0,1] =  1.0
  weight_np[0,2,0,1] =  angle_norm
  weight_np[2,0,0,1] = -angle_norm
  weight_np[2,1,0,1] = -1.0
  weight_np[2,2,0,1] = -angle_norm
  weight_np[0,0,0,0] =  angle_norm
  weight_np[1,0,0,0] =  1.0
  weight_np[2,0,0,0] =  angle_norm
  weight_np[0,2,0,0] = -angle_norm
  weight_np[1,2,0,0] = -1.0
  weight_np[2,2,0,0] = -angle_norm
  weight = tf.constant(np.float32(weight_np))

  # calc gradientes
  pressure_integral = _simple_conv_2d(boundary, weight)[:,2:-2,2:-2]
  pressure_field = pressure_field[:,3:-3,3:-3]
  force = ((1.0 - boundary[:,3:-3,3:-3]) * pressure_integral) * pressure_field
  return force

def calc_force_3d(boundary, pressure_field):
  # implementation of spatial divergence
  # reimplemented from torch FluidNet implementation
  print(boundary.get_shape())
  print(pressure_field.get_shape())

  # make weight for x divergence
  weight_np = np.zeros([3,3,3,1,3])
  angle_norm_3 = 1.0 / np.sqrt(3.0)
  angle_norm_2 = 1.0 / np.sqrt(2.0)
  weight_np[0,0,0,0,0] =   angle_norm_3
  weight_np[1,0,0,0,0] =   angle_norm_2
  weight_np[2,0,0,0,0] =   angle_norm_3
  weight_np[0,1,0,0,0] =   angle_norm_2
  weight_np[1,1,0,0,0] =   1.0
  weight_np[2,1,0,0,0] =   angle_norm_2
  weight_np[0,2,0,0,0] =   angle_norm_3
  weight_np[1,2,0,0,0] =   angle_norm_2
  weight_np[2,2,0,0,0] =   angle_norm_3
  weight_np[0,0,2,0,0] =  -angle_norm_3
  weight_np[1,0,2,0,0] =  -angle_norm_2
  weight_np[2,0,2,0,0] =  -angle_norm_3
  weight_np[0,1,2,0,0] =  -angle_norm_2
  weight_np[1,1,2,0,0] =  -1.0
  weight_np[2,1,2,0,0] =  -angle_norm_2
  weight_np[0,2,2,0,0] =  -angle_norm_3
  weight_np[1,2,2,0,0] =  -angle_norm_2
  weight_np[2,2,2,0,0] =  -angle_norm_3

  weight_np[0,0,0,0,1] =   angle_norm_3
  weight_np[1,0,0,0,1] =   angle_norm_2
  weight_np[2,0,0,0,1] =   angle_norm_3
  weight_np[0,0,1,0,1] =   angle_norm_2
  weight_np[1,0,1,0,1] =   1.0
  weight_np[2,0,1,0,1] =   angle_norm_2
  weight_np[0,0,2,0,1] =   angle_norm_3
  weight_np[1,0,2,0,1] =   angle_norm_2
  weight_np[2,0,2,0,1] =   angle_norm_3
  weight_np[0,2,0,0,1] =  -angle_norm_3
  weight_np[1,2,0,0,1] =  -angle_norm_2
  weight_np[2,2,0,0,1] =  -angle_norm_3
  weight_np[0,2,1,0,1] =  -angle_norm_2
  weight_np[1,2,1,0,1] =  -1.0
  weight_np[2,2,1,0,1] =  -angle_norm_2
  weight_np[0,2,2,0,1] =  -angle_norm_3
  weight_np[1,2,2,0,1] =  -angle_norm_2
  weight_np[2,2,2,0,1] =  -angle_norm_3

  weight_np[0,0,0,0,2] =   angle_norm_3
  weight_np[0,1,0,0,2] =   angle_norm_2
  weight_np[0,2,0,0,2] =   angle_norm_3
  weight_np[0,0,1,0,2] =   angle_norm_2
  weight_np[0,1,1,0,2] =   1.0
  weight_np[0,2,1,0,2] =   angle_norm_2
  weight_np[0,0,2,0,2] =   angle_norm_3
  weight_np[0,1,2,0,2] =   angle_norm_2
  weight_np[0,2,2,0,2] =   angle_norm_3
  weight_np[2,0,0,0,2] =  -angle_norm_3
  weight_np[2,1,0,0,2] =  -angle_norm_2
  weight_np[2,2,0,0,2] =  -angle_norm_3
  weight_np[2,0,1,0,2] =  -angle_norm_2
  weight_np[2,1,1,0,2] =  -1.0
  weight_np[2,2,1,0,2] =  -angle_norm_2
  weight_np[2,0,2,0,2] =  -angle_norm_3
  weight_np[2,1,2,0,2] =  -angle_norm_2
  weight_np[2,2,2,0,2] =  -angle_norm_3


  weight = tf.constant(np.float32(weight_np))

  # calc gradientes
  pressure_integral = _simple_conv_3d(boundary, weight)[:,3:-3,3:-3,3:-3]
  pressure_field = pressure_field[:,3:-3,3:-3,3:-3]
  force = ((1.0 - boundary[:,3:-3,3:-3,3:-3]) * pressure_integral) * pressure_field
  return force

