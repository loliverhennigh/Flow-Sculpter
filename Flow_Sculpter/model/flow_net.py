
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import network_architecture
import inputs.flow_input as flow_input
import model.loss as loss
import utils.boundary_utils as boundary_utils
import model.nn as nn
from model.image_summary import *

FLAGS = tf.app.flags.FLAGS

# Constants describing the training process and model.

# Training params
tf.app.flags.DEFINE_string('base_dir_flow', '../checkpoints_flow',
                            """dir to store trained flow net """)
tf.app.flags.DEFINE_string('base_dir_boundary', '../checkpoints_boundary',
                            """dir to store trained net boundary """)
tf.app.flags.DEFINE_integer('batch_size', 4,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('nr_gpus', 1,
                           """ number of gpus for training (each gpu with have batch size FLAGS.batch_size""")
tf.app.flags.DEFINE_integer('max_steps',  3000000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_float('keep_prob', 1.0,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('lr', 1e-4,
                            """ r dropout """)
tf.app.flags.DEFINE_string('shape', '256x256',
                            """ shape of flow """)
tf.app.flags.DEFINE_integer('dims', 2,
                            """ dims of flow """)
tf.app.flags.DEFINE_integer('obj_size', 128,
                            """ max size of voxel object """)


# model params flow
tf.app.flags.DEFINE_string('flow_model', 'residual_network',
                           """ model name to train """)
tf.app.flags.DEFINE_integer('filter_size', 8,
                           """ filter size of first res block (preceding layers have double the filter size) """)
tf.app.flags.DEFINE_integer('nr_downsamples', 6,
                           """ number of downsamples in u network """)
tf.app.flags.DEFINE_integer('nr_residual_blocks', 3,
                           """ number of res blocks after each downsample """)
tf.app.flags.DEFINE_bool('gated', True,
                           """ gated resnet or not """)
tf.app.flags.DEFINE_string('nonlinearity', 'relu',
                           """ nonlinearity used such as concat_elu, elu, concat_relu, relu """)
tf.app.flags.DEFINE_bool('sdf', False,
                           """ whether to use the signed distance function on inputs """)

# model params boundary
tf.app.flags.DEFINE_string('boundary_model', 'fc_conv',
                           """ model name to train boundary network on """)
tf.app.flags.DEFINE_integer('nr_boundary_params', 46,
                            """ number of boundary paramiters """)

# params boundary learn
tf.app.flags.DEFINE_string('boundary_learn_loss', "drag_xy",
                            """ what to mimimize in the boundary learning stuff """)
tf.app.flags.DEFINE_float('boundary_learn_lr', 0.001,
                            """ learning rate when learning boundary """)
tf.app.flags.DEFINE_integer('boundary_learn_steps', 500,
                            """ number of steps when learning boundary """)

# test params
tf.app.flags.DEFINE_string('test_set', "car",
                            """ either car or random """)

def inputs_flow(batch_size, shape, dims):
  """makes input vector
  Return:
    x: input vector, may be filled 
  """
  boundary  = tf.placeholder(tf.float32, [batch_size] + shape + [1])
  true_flow = tf.placeholder(tf.float32, [batch_size] + shape + [dims+1])
  image_summary('boundarys', boundary)
  return boundary, true_flow

def inputs_boundary(input_dims, batch_size, shape):
  """makes input vector
  Return:
    x: input vector, may be filled 
  """
  inputs = tf.placeholder(tf.float32, [batch_size] + [input_dims])
  boundary = tf.placeholder(tf.float32, [batch_size] + shape + [1])
  image_summary('boundarys', boundary)
  return inputs, boundary

def inputs_boundary_learn(batch_size=1, set_params=None, set_params_pos=None, noise_std=None):
  
  # make params
  params_op_set = tf.placeholder(tf.float32, [batch_size, FLAGS.nr_boundary_params])
  params_op = tf.Variable(np.zeros((batch_size, FLAGS.nr_boundary_params)).astype(dtype=np.float32), name="params")
  params_op_init = tf.group(params_op.assign(params_op_set))

  # make squeeze loss to keep params almost between -0.5 and 0.5
  squeeze_loss = tf.abs(params_op) - .45
  squeeze_loss = tf.reduce_sum(tf.maximum(squeeze_loss, 0.0))

  # Now bound params between 0.0 and 1.0
  params_op = nn.hard_sigmoid(params_op)

  # Now bound params to appropriote variable ranges
  params_range_lower, params_range_upper = boundary_utils.get_params_range(FLAGS.nr_boundary_params, FLAGS.dims)
  params_range_upper = np.expand_dims(params_range_upper, axis=0)
  params_range_lower = np.expand_dims(params_range_lower, axis=0)
  params_range_upper = params_range_upper - params_range_lower
  params_op = (params_op * tf.constant(params_range_upper, dtype=tf.float32)) + tf.constant(params_range_lower, dtype=tf.float32)

  # Now add possibly add noise
  if noise_std is not None:
    noise = tf.random_normal(shape=tf.shape(params_op), mean=0.0, stddev=noise_std, dtype=tf.float32) 
    params_op += noise

  # Now hard set values
  if set_params is not None:
    params_op = tf.split(params_op, batch_size, axis=0)
    params_op_store = []
    for par in params_op:
      for i in xrange(set_params.shape[0]):
        params_op_store.append(par)
    params_op = tf.concat(params_op_store, axis=0)
    set_params     = np.concatenate(batch_size * [set_params], axis=0)
    set_params_pos = np.concatenate(batch_size * [set_params_pos], axis=0)
    params_op = (set_params_pos * params_op) + set_params

  return params_op, params_op_init, params_op_set, squeeze_loss

def feed_dict_boundary(input_dims, batch_size, shape, set_params=None):
  #input_batch, boundary_batch = boundary_utils.wing_boundary_batch_2d(input_dims, batch_size, shape, set_angle=set_angle)
  input_batch, boundary_batch = boundary_utils.wing_boundary_batch(input_dims, batch_size, shape, FLAGS.dims)
  return input_batch, boundary_batch

def inference_flow(boundary, keep_prob=FLAGS.keep_prob):
  """Builds network.
  Args:
    inputs: input to network 
    keep_prob: dropout layer
  """
  with tf.variable_scope("flow_network") as scope:
    if FLAGS.flow_model == 'residual_network':
      predicted_flow = network_architecture.res_u_template(boundary, 
                                                           keep_prob=1.0,
                                                           filter_size=FLAGS.filter_size,
                                                           nr_downsamples=FLAGS.nr_downsamples,
                                                           nr_residual_blocks=FLAGS.nr_residual_blocks,
                                                           gated=FLAGS.gated, 
                                                           nonlinearity=FLAGS.nonlinearity)
    if FLAGS.flow_model == 'xiao_network':
      predicted_flow = network_architecture.xiao_template(boundary)
  return predicted_flow

def inference_boundary(batch_size, shape, inputs, full_shape=None):
  with tf.variable_scope("boundary_network") as scope:
    boundary_gen = network_architecture.res_generator_template(batch_size, shape, inputs, full_shape)
  return boundary_gen

def loss_flow(true_flow, predicted_flow):
  loss_total = 0 
  loss_mse  = tf.nn.l2_loss(true_flow - predicted_flow)/(FLAGS.batch_size*FLAGS.nr_gpus)
  loss_grad = loss.loss_gradient_difference(true_flow, predicted_flow)/(FLAGS.batch_size*FLAGS.nr_gpus)
  #loss_total = loss_mse + loss_grad
  loss_total = loss_mse

  # image summary
  difference_i = tf.abs(true_flow - predicted_flow)
  difference_i = tf.expand_dims(tf.reduce_sum(difference_i, axis=3), axis=3)
  image_summary('difference', difference_i)
  image_summary('flow_x_true', true_flow[...,0:1])
  image_summary('flow_x_predicted', predicted_flow[...,0:1])
  image_summary('flow_y_true',  true_flow[...,1:2])
  image_summary('flow_y_predicted', predicted_flow[...,1:2])
  if len(true_flow.get_shape()) == 5:
    image_summary('flow_z_true',  true_flow[...,2:3])
    image_summary('flow_z_predicted', predicted_flow[...,2:3])
  image_summary('flow_p_true',  true_flow[...,-1:])
  image_summary('flow_p_predicted', predicted_flow[...,-1:])

  # loss summary
  with tf.device('/cpu:0'):
    tf.summary.scalar('loss_mse', loss_mse)
    tf.summary.scalar('loss_grad', loss_grad)

  return loss_total

def loss_boundary(true_boundary, generated_boundary):
  #intersection = tf.reduce_sum(generated_boundary * true_boundary)
  #loss_dice = -(2. * intersection + 1.) / (tf.reduce_sum(true_boundary) + tf.reduce_sum(generated_boundary) + 1.)
  #boundary_shape = nn.int_shape(generated_boundary) 
  #loss_grad = loss.loss_gradient_difference(true_boundary, generated_boundary)/(np.prod(np.array(boundary_shape[1:-1])))
  #loss_total = 0.5*loss_grad + loss_dice
  #loss_total = loss_dice
  loss_total = tf.nn.l2_loss(true_boundary - generated_boundary)

  # image summary
  image_summary('boundary_predicted', generated_boundary)
  image_summary('boundary_diff', tf.abs(true_boundary - generated_boundary))

  # loss summary
  #tf.summary.scalar('loss_dice', loss_dice)
  #tf.summary.scalar('loss_grad', loss_grad)
  tf.summary.scalar('loss_total', loss_total)
  return loss_total

def train(total_loss, lr, train_type="flow_network", global_step=None, variables=None):
   if train_type == "flow_network" or train_type == "boundary_network":
     train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step)
   elif train_type == "boundary_params":
     train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(total_loss, var_list=variables)
     #train_op = tf.train.GradientDescentOptimizer(lr).minimize(total_loss, var_list=variables)
   return train_op

