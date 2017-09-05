
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import network_architecture
import inputs.flow_input as flow_input
import utils.boundary_utils as boundary_utils

FLAGS = tf.app.flags.FLAGS

# Constants describing the training process and model.

# Training params
tf.app.flags.DEFINE_string('base_dir_flow', '../checkpoints_flow',
                            """dir to store trained flow net """)
tf.app.flags.DEFINE_string('base_dir_boundary', '../checkpoints_boundary',
                            """dir to store trained net boundary """)
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('nr_gpus', 1,
                           """ number of gpus for training (each gpu with have batch size FLAGS.batch_size""")
tf.app.flags.DEFINE_integer('max_steps',  3000000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_float('keep_prob', 1.0,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('lr', 1e-4,
                            """ r dropout """)
tf.app.flags.DEFINE_string('shape', '256x1024',
                            """ shape of flow """)
tf.app.flags.DEFINE_integer('dims', 2,
                            """ dims of flow """)
tf.app.flags.DEFINE_integer('obj_size', 128,
                            """ max size of voxel object """)


# model params flow
tf.app.flags.DEFINE_string('flow_model', 'residual_network',
                           """ model name to train """)
tf.app.flags.DEFINE_integer('nr_pyramids', 0,
                           """ number of iteration in increaseing size resolution """)
tf.app.flags.DEFINE_integer('filter_size', 8,
                           """ filter size of first res block (preceding layers have double the filter size) """)
tf.app.flags.DEFINE_integer('nr_downsamples', 6,
                           """ number of downsamples in u network """)
tf.app.flags.DEFINE_integer('nr_residual_blocks', 2,
                           """ number of res blocks after each downsample """)
tf.app.flags.DEFINE_bool('gated', True,
                           """ gated resnet or not """)
tf.app.flags.DEFINE_string('nonlinearity', 'concat_elu',
                           """ nonlinearity used such as concat_elu, elu, concat_relu, relu """)

# model params boundary
tf.app.flags.DEFINE_string('boundary_type', 'shapes',
                           """ type of boundarys to train on """)
tf.app.flags.DEFINE_string('boundary_model', 'fc_conv',
                           """ model name to train boundary network on """)
tf.app.flags.DEFINE_integer('nr_boundary_params', 12,
                            """ number of boundary paramiters """)
tf.app.flags.DEFINE_float('beta', 0.1,
                            """ learning rate when learning boundary """)

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
  with tf.device('/cpu:0'):
    tf.summary.image('boundarys', boundary)
  return boundary, true_flow

def inputs_boundary(input_dims, batch_size, shape):
  """makes input vector
  Return:
    x: input vector, may be filled 
  """
  inputs = tf.placeholder(tf.float32, [batch_size] + [input_dims])
  boundary = tf.placeholder(tf.float32, [batch_size] + shape + [1])
  tf.summary.image('boundarys', boundary)
  return inputs, boundary

def inputs_boundary_learn(batch_size=1, set_angle=None):
  if set_angle is not None:
    params_op_set = tf.placeholder(tf.float32, [batch_size, FLAGS.nr_boundary_params-1])
    params_op = tf.Variable(np.zeros((batch_size, FLAGS.nr_boundary_params-1)).astype(dtype=np.float32), name="params")
    params_op_init = tf.group(params_op.assign(params_op_set))
    noise = tf.random_normal(shape=tf.shape(params_op), mean=0.0, stddev=0.001, dtype=tf.float32) 
    params_op += noise
    params_op = (tf.sigmoid(params_op) - tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])) / tf.constant([1.0, 0.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    angles = tf.constant(set_angle, dtype=tf.float32)
    angles = tf.reshape(tf.concat(batch_size*[angles], axis=0), [batch_size*len(set_angle),1])
    params_op = tf.concat(len(set_angle)*[params_op], axis=0)
    params_op = tf.concat([angles, params_op], axis = 1)
  else:
    params_op_set = tf.placeholder(tf.float32, [batch_size, FLAGS.nr_boundary_params])
    params_op = tf.Variable(np.zeros((batch_size, FLAGS.nr_boundary_params)).astype(dtype=np.float32), name="params")
    params_op_init = tf.group(params_op.assign(params_op_set))
  return params_op, params_op_init, params_op_set

def feed_dict_flows(batch_size, shape):
  boundarys = []
  us = []
  u_ons = []
  boundary_creator = boundary_utils.get_boundary_creator(FLAGS.boundary_type)
  inflow_vector = inflow.get_inflow_vector(FLAGS.inflow_type)
  for i in xrange(batch_size):
    boundarys.append(boundary_creator(shape))
    u, u_on = inflow_vector(shape, np.array([FLAGS.inflow_value, 0.0, 0.0]))
    us.append(u)
    u_ons.append(u_on)
  boundarys = np.concatenate(boundarys)
  us = np.concatenate(us)
  u_ons = np.concatenate(u_ons)
  boundarys = np.concatenate([boundarys,us,u_ons], axis=-1)
  return boundarys

def feed_dict_boundary(input_dims, batch_size, shape):
  input_batch, boundary_batch = boundary_utils.wing_boundary_batch_2d(input_dims, batch_size, shape)
  return input_batch, boundary_batch

def inference_flow(boundary, true_flow, keep_prob=FLAGS.keep_prob, nr_pyramids=FLAGS.nr_pyramids):
  """Builds network.
  Args:
    inputs: input to network 
    keep_prob: dropout layer
  """
  with tf.variable_scope("flow_network") as scope:
    pyr_true_flow, pyr_pred_flow = network_architecture.pyramid_net(boundary, true_flow, 
                                                                    keep_prob=keep_prob,
                                                                    filter_size=FLAGS.filter_size,
                                                                    nr_pyramids=nr_pyramids,
                                                                    nr_downsamples=FLAGS.nr_downsamples,
                                                                    nr_residual_blocks=FLAGS.nr_residual_blocks,
                                                                    gated=FLAGS.gated, 
                                                                    nonlinearity=FLAGS.nonlinearity)
  return pyr_true_flow, pyr_pred_flow

def inference_boundary(boundary, shape):
  with tf.variable_scope("boundary_network") as scope:
    mean, stddev, x_sampled = network_architecture.res_VAE_encoding(boundary, hidden_size=FLAGS.nr_boundary_params)
    boundary_prime = network_architecture.res_VAE_decoding(x_sampled, shape)
  return mean, stddev, x_sampled, boundary_prime

def inference_boundary_decoder(hidden_state, shape):
  with tf.variable_scope("boundary_network") as scope:
    boundary_prime = network_architecture.res_VAE_decoding(hidden_state, shape)
  return boundary_prime

def inference_boundary_generator(batch_size, shape, inputs=None, full_shape=None):
  with tf.variable_scope("boundary_network_generator") as scope:
    boundary_gen = network_architecture.res_generator_template(batch_size, shape, inputs, full_shape)
  return boundary_gen

def inference_boundary_discriminator(boundary):
  with tf.variable_scope("boundary_network_discriminator") as scope:
    is_boundary = network_architecture.res_discriminator_template(boundary)
  return is_boundary

def loss_flow(pyramid_true_flow, pyramid_predicted_flow):
 
  loss = 0 
  i = 0
  for true_flow, predicted_flow in zip(pyramid_true_flow, pyramid_predicted_flow):
    loss += tf.nn.l2_loss(true_flow - predicted_flow)/(FLAGS.batch_size*FLAGS.nr_gpus)

    # image summary
    with tf.device('/cpu:0'):
      difference_i = tf.abs(true_flow - predicted_flow)
      difference_i = tf.expand_dims(tf.reduce_sum(difference_i, axis=3), axis=3)
      tf.summary.image('difference_i' + str(i), difference_i)
      tf.summary.image('flow_x_true_i' + str(i), true_flow[:,:,:,0:1])
      tf.summary.image('flow_x_predicted_i' + str(i), predicted_flow[:,:,:,0:1])
      tf.summary.image('flow_y_true_i' + str(i),  true_flow[:,:,:,1:2])
      tf.summary.image('flow_y_predicted_i' + str(i), predicted_flow[:,:,:,1:2])
      tf.summary.image('flow_p_true_i' + str(i),  true_flow[:,:,:,2:3])
      tf.summary.image('flow_p_predicted_i' + str(i), predicted_flow[:,:,:,2:3])

    # loss summary
    with tf.device('/cpu:0'):
      tf.summary.scalar('pyramid_loss_' + str(i), loss)

    # iterate ind
    i += 1

  return loss


def loss_boundary(true_boundary, generated_boundary):
  intersection = tf.reduce_sum(generated_boundary * true_boundary)
  loss = -(2. * intersection + 1.) / (tf.reduce_sum(true_boundary) + tf.reduce_sum(generated_boundary) + 1.)
  tf.summary.scalar('loss', loss)
  tf.summary.image('boundary_predicted', generated_boundary)
  return loss

"""
def loss_boundary(mean, stddev, true_boundary, predicted_boundary):
  # epsilon 
  epsilon = 1e-10

  # calc loss from vae
  kl_loss = 0.5 * (tf.square(mean) + tf.square(stddev) 
                   - 2.0 * tf.log(stddev + epsilon) - 1.0)
  loss_vae = FLAGS.beta * tf.reduce_sum(kl_loss)

  # log loss for reconstruction
  loss_reconstruction = tf.reduce_sum(-true_boundary * tf.log(predicted_boundary + epsilon) -
                (1.0 - true_boundary) * tf.log(1.0 - predicted_boundary + epsilon)) 

  # save for tensorboard
  tf.summary.scalar('loss_vae', loss_vae)
  tf.summary.scalar('loss_reconstruction', loss_reconstruction)
  tf.summary.image('boundary_true', true_boundary)
  tf.summary.image('boundary_predicted', predicted_boundary)

  # calc total loss 
  loss = tf.reduce_sum(loss_vae + loss_reconstruction)
  #loss = loss_reconstruction
  #loss = tf.nn.l2_loss(true_boundary - predicted_boundary)
  return loss
"""

def train(total_loss, lr, train_type="flow_network", global_step=None, variables=None):
   if train_type == "flow_network" or train_type == "boundary_network":
     train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step)
   elif train_type == "boundary_params":
     train_op = tf.train.GradientDescentOptimizer(lr).minimize(total_loss, var_list=variables)
   return train_op

