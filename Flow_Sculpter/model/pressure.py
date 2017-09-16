
import tensorflow as tf
import numpy as np

def _simple_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y

def _simple_conv_3d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv3d(x, k, [1, 1, 1, 1, 1], padding='SAME')
  return y

def force_2d(boundary, pressure_field):
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
  pressure_integral = _simple_conv_2d(boundary, weight)[:,3:-3,3:-3]
  pressure_field = pressure_field[:,3:-3,3:-3]
  force = ((1.0 - boundary[:,3:-3,3:-3]) * pressure_integral) * pressure_field
  return force

def force_3d(boundary, pressure_field):
  # implementation of spatial divergence
  # reimplemented from torch FluidNet implementation

  # make weight for x divergence
  weight_np = np.zeros([3,3,3,1,2])
  angle_norm_3 = 1.0 / np.sqrt(3.0)
  angle_norm_2 = 1.0 / np.sqrt(2.0)
  weight_np[0,0,0,0,1] =   angle_norm_3
  weight_np[1,0,0,0,1] =   angle_norm_2
  weight_np[2,0,0,0,1] =   angle_norm_3
  weight_np[0,1,0,0,1] =   angle_norm_2
  weight_np[1,1,0,0,1] =   1.0
  weight_np[2,1,0,0,1] =   angle_norm_2
  weight_np[0,2,0,0,1] =   angle_norm_3
  weight_np[1,2,0,0,1] =   angle_norm_2
  weight_np[2,2,0,0,1] =   angle_norm_3
  weight_np[0,0,2,0,1] =  -angle_norm_3
  weight_np[1,0,2,0,1] =  -angle_norm_2
  weight_np[2,0,2,0,1] =  -angle_norm_3
  weight_np[0,1,2,0,1] =  -angle_norm_2
  weight_np[1,1,2,0,1] =  -1.0
  weight_np[2,1,2,0,1] =  -angle_norm_2
  weight_np[0,2,2,0,1] =  -angle_norm_3
  weight_np[1,2,2,0,1] =  -angle_norm_2
  weight_np[2,2,2,0,1] =  -angle_norm_3

  weight_np[0,0,0,0,2] =   angle_norm_3
  weight_np[1,0,0,0,2] =   angle_norm_2
  weight_np[2,0,0,0,2] =   angle_norm_3
  weight_np[0,0,1,0,2] =   angle_norm_2
  weight_np[1,0,1,0,2] =   1.0
  weight_np[2,0,1,0,2] =   angle_norm_2
  weight_np[0,0,2,0,2] =   angle_norm_3
  weight_np[1,0,2,0,2] =   angle_norm_2
  weight_np[2,0,2,0,2] =   angle_norm_3
  weight_np[0,2,0,0,2] =  -angle_norm_3
  weight_np[1,2,0,0,2] =  -angle_norm_2
  weight_np[2,2,0,0,2] =  -angle_norm_3
  weight_np[0,2,1,0,2] =  -angle_norm_2
  weight_np[1,2,1,0,2] =  -1.0
  weight_np[2,2,1,0,2] =  -angle_norm_2
  weight_np[0,2,2,0,2] =  -angle_norm_3
  weight_np[1,2,2,0,2] =  -angle_norm_2
  weight_np[2,2,2,0,2] =  -angle_norm_3

  weight_np[0,0,0,0,1] =   angle_norm_3
  weight_np[0,1,0,0,1] =   angle_norm_2
  weight_np[0,2,0,0,1] =   angle_norm_3
  weight_np[0,0,1,0,1] =   angle_norm_2
  weight_np[0,1,1,0,1] =   1.0
  weight_np[0,2,1,0,1] =   angle_norm_2
  weight_np[0,0,2,0,1] =   angle_norm_3
  weight_np[0,1,2,0,1] =   angle_norm_2
  weight_np[0,2,2,0,1] =   angle_norm_3
  weight_np[2,0,0,0,1] =  -angle_norm_3
  weight_np[2,1,0,0,1] =  -angle_norm_2
  weight_np[2,2,0,0,1] =  -angle_norm_3
  weight_np[2,0,1,0,1] =  -angle_norm_2
  weight_np[2,1,1,0,1] =  -1.0
  weight_np[2,2,1,0,1] =  -angle_norm_2
  weight_np[2,0,2,0,1] =  -angle_norm_3
  weight_np[2,1,2,0,1] =  -angle_norm_2
  weight_np[2,2,2,0,1] =  -angle_norm_3


  weight = tf.constant(np.float32(weight_np))

  # calc gradientes
  pressure_integral = _simple_conv_2d(boundary, weight)[:,3:-3,3:-3]
  pressure_field = pressure_field[:,3:-3,3:-3]
  force = ((1.0 - boundary[:,3:-3,3:-3]) * pressure_integral) * pressure_field
  return force


def spatial_divergence_3d(field):
  # implementation of spatial divergence
  # reimplemented from torch FluidNet implementation
  field_shape = int_shape(field)
  field = tf.reshape(field, [field_shape[0]*field_shape[1], field_shape[2], field_shape[3], field_shape[4], field_shape[5]])

  # make weight for x divergence
  weight_x_np = np.zeros([3,1,1,4,1])
  weight_x_np[0,0,0,0,0] = -1.0/2.0
  weight_x_np[1,0,0,0,0] = 0.0 
  weight_x_np[2,0,0,0,0] = 1.0/2.0

  weight_x = tf.constant(np.float32(weight_x_np))

  # make weight for y divergence
  weight_y_np = np.zeros([1,3,1,4,1])
  weight_y_np[0,0,0,1,0] = -1.0/2.0
  weight_y_np[0,1,0,1,0] = 0.0 
  weight_y_np[0,2,0,1,0] = 1.0/2.0

  weight_y = tf.constant(np.float32(weight_y_np))

  # make weight for z divergence
  weight_z_np = np.zeros([1,1,3,4,1])
  weight_z_np[0,0,0,2,0] = -1.0/2.0
  weight_z_np[0,0,1,2,0] = 0.0 
  weight_z_np[0,0,2,2,0] = 1.0/2.0

  weight_z = tf.constant(np.float32(weight_z_np))

  # calc gradientes
  field_dx = _simple_conv_3d(field, weight_x)
  field_dy = _simple_conv_3d(field, weight_y)
  field_dz = _simple_conv_3d(field, weight_z)

  # divergence of field
  field_div = field_dx + field_dy + field_dz

  # kill boundarys (this is not correct! I should use boundarys but for right now I will not)
  field_div = tf.abs(field_div[:,1:-2,1:-2,:])

  return field_div

