
"""functions used to construct different losses
"""

import tensorflow as tf
import numpy as np
from divergence import *
from pressure import calc_force


FLAGS = tf.app.flags.FLAGS


def loss_mse(true, generated):
  loss = tf.nn.l2_loss(true - generated)
  return loss

def loss_pressure_difference(true, generated, boundary):
  true_force = calc_force(boundary, true[...,-1:])
  generated_force = calc_force(boundary, generated[...,-1:])
  loss = tf.nn.l2_loss(true_force - generated_force)
  return loss

def loss_divergence(true_field, generated_field):
  if len(true_field.get_shape()) == 5:
    true_field_div = spatial_divergence_2d(true_field)
    generated_field_div = spatial_divergence_2d(generated_field)
  if len(true_field.get_shape()) == 6:
    true_field_div = spatial_divergence_3d(true_field)
    generated_field_div = spatial_divergence_3d(generated_field)
  loss = tf.abs(tf.nn.l2_loss(true_field_div) - tf.nn.l2_loss(generated_field_div))
  return loss

def loss_gradient_difference(true, generated):
  # seen in here https://arxiv.org/abs/1511.05440
  if len(true.get_shape()) == 4:
    true_x_shifted_right = true[:,1:,:,:]
    true_x_shifted_left = true[:,:-1,:,:]
    true_x_gradient = tf.abs(true_x_shifted_right - true_x_shifted_left)

    generated_x_shifted_right = generated[:,1:,:,:]
    generated_x_shifted_left = generated[:,:-1,:,:]
    generated_x_gradient = tf.abs(generated_x_shifted_right - generated_x_shifted_left)

    loss_x_gradient = tf.nn.l2_loss(true_x_gradient - generated_x_gradient)

    true_y_shifted_right = true[:,:,1:,:]
    true_y_shifted_left = true[:,:,:-1,:]
    true_y_gradient = tf.abs(true_y_shifted_right - true_y_shifted_left)

    generated_y_shifted_right = generated[:,:,1:,:]
    generated_y_shifted_left = generated[:,:,:-1,:]
    generated_y_gradient = tf.abs(generated_y_shifted_right - generated_y_shifted_left)
    
    loss_y_gradient = tf.nn.l2_loss(true_y_gradient - generated_y_gradient)

    loss = loss_x_gradient + loss_y_gradient

  else:
    true_x_shifted_right = true[:,1:,:,:,:]
    true_x_shifted_left = true[:,:-1,:,:,:]
    true_x_gradient = tf.abs(true_x_shifted_right - true_x_shifted_left)

    generated_x_shifted_right = generated[:,1:,:,:,:]
    generated_x_shifted_left = generated[:,:-1,:,:,:]
    generated_x_gradient = tf.abs(generated_x_shifted_right - generated_x_shifted_left)

    loss_x_gradient = tf.nn.l2_loss(true_x_gradient - generated_x_gradient)

    true_y_shifted_right = true[:,:,1:,:,:]
    true_y_shifted_left = true[:,:,:-1,:,:]
    true_y_gradient = tf.abs(true_y_shifted_right - true_y_shifted_left)

    generated_y_shifted_right = generated[:,:,1:,:,:]
    generated_y_shifted_left = generated[:,:,:-1,:,:]
    generated_y_gradient = tf.abs(generated_y_shifted_right - generated_y_shifted_left)
    
    loss_y_gradient = tf.nn.l2_loss(true_y_gradient - generated_y_gradient)

    true_z_shifted_right = true[:,:,:,1:,:]
    true_z_shifted_left = true[:,:,:,:-1,:]
    true_z_gradient = tf.abs(true_z_shifted_right - true_z_shifted_left)

    generated_z_shifted_right = generated[:,:,:,1:,:]
    generated_z_shifted_left = generated[:,:,:,:-1,:]
    generated_z_gradient = tf.abs(generated_z_shifted_right - generated_z_shifted_left)
    
    loss_z_gradient = tf.nn.l2_loss(true_z_gradient - generated_z_gradient)

    loss = loss_x_gradient + loss_y_gradient + loss_z_gradient

  return loss

