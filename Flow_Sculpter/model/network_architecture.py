
import tensorflow as tf
import numpy as np
import nn

def pyramid_net(boundary, true_flow, keep_prob=1.0, filter_size=8, nr_pyramids=1, nr_downsamples=2, nr_residual_blocks=2, gated=True, nonlinearity='concat_elu'):

  # define the pieces of the network

  def mini_res_u_network(inputs, keep_prob, filter_size, nr_downsamples, nr_residual_blocks, gated, nonlinearity):
    # store for as
    a = []
    # set nonlinearity
    nonlinearity = nn.set_nonlinearity(nonlinearity)
    # encoding piece
    x_i = inputs
    for i in xrange(nr_downsamples):
      for j in xrange(nr_residual_blocks):
        x_i = nn.res_block(x_i, filter_size=filter_size, keep_p=keep_prob, gated=gated, nonlinearity=nonlinearity, name="res_encode_" + str(i) + "_block_" + str(j))
      if i < nr_downsamples-1:
        a.append(x_i)
        filter_size = filter_size * 2
        x_i = nn.res_block(x_i, filter_size=filter_size, keep_p=keep_prob, gated=gated, nonlinearity=nonlinearity, stride=2, name="res_encode_" + str(i) + "_block_" + str(nr_residual_blocks))
    # decoding piece
    for i in xrange(nr_downsamples-1):
      filter_size = filter_size / 2
      x_i = nn.transpose_conv_layer(x_i, 4, 2, filter_size, "up_conv_" + str(i))
      x_i = nn.res_block(x_i, a=a.pop(), filter_size=filter_size, keep_p=keep_prob, gated=gated, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_0")
      for j in xrange(nr_residual_blocks-1):
        x_i = nn.res_block(x_i, filter_size=filter_size, keep_p=keep_prob, gated=gated, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_" + str(j+1))
    x_i = nn.conv_layer(x_i, 3, 1, 3, "final_conv")
    return x_i

  # mini res u net template
  mini_res_u_template = tf.make_template('mini_res_u_template', mini_res_u_network)

  def upsampleing_res_u_network(inputs, filter_size, nr_residual_blocks, gated, nonlinearity):
    # res_1
    x_i = inputs
    # set nonlinearity
    nonlinearity = nn.set_nonlinearity(nonlinearity)
    # upsampling
    for i in xrange(nr_residual_blocks):
      x_i = nn.res_block(x_i, filter_size=filter_size, keep_p=keep_prob, gated=gated, nonlinearity=nonlinearity, name="res_block_" + str(i), begin_nonlinearity=True)
    x_i = nn.transpose_conv_layer(x_i, 4, 2, filter_size, "up_conv_" + str(i))
    return x_i
 
  # upsampleing res u network
  if nr_pyramids > 0:
    upsampleing_res_u_template      = tf.make_template('upsampleing_res_u_template', upsampleing_res_u_network)
  upsampleing_res_u_fake_template = tf.make_template('upsampleing_res_u_fake_template', upsampleing_res_u_network)

  # generate list of resized inputs
  pyramid_boundary = []
  pyramid_true_flow = []
  pyramid_boundary.append(boundary)
  pyramid_true_flow.append(true_flow)
  shape = nn.int_shape(boundary)[1:-1]
  for i in xrange(nr_pyramids):
    print(pyramid_boundary[i])
    pyramid_boundary.append(tf.nn.max_pool(pyramid_boundary[i], ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID"))
    pyramid_true_flow.append(tf.nn.avg_pool(pyramid_true_flow[i], ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID"))
  pyramid_boundary = list(reversed(pyramid_boundary))
  pyramid_true_flow = list(reversed(pyramid_true_flow))

  pyramid_predicted_flow = []
  for i in xrange(nr_pyramids+1):
    # get current boundary resolution
    boundary_i = pyramid_boundary[i]
    # concat previous upsampled flow
    if i == 0:
      zeros_flow = upsampleing_res_u_fake_template(tf.nn.max_pool(boundary_i, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID"), filter_size, nr_residual_blocks, gated, nonlinearity)
      boundary_i = tf.concat([boundary_i, zeros_flow], axis=3)
    else:
      boundary_i = tf.concat([boundary_i, upsampled_flow_i], axis=3)
    # run through mini res u network
    flow_i = mini_res_u_template(boundary_i, keep_prob, filter_size, nr_downsamples, nr_residual_blocks, gated, nonlinearity)
    pyramid_predicted_flow.append(flow_i)
    # run through upsampling network
    if i != nr_pyramids:
      upsampled_flow_i = upsampleing_res_u_template(flow_i, filter_size, nr_residual_blocks, gated, nonlinearity)

  return pyramid_true_flow, pyramid_predicted_flow

def conv_res(inputs, true_flow, nr_res_blocks=2, keep_prob=1.0, nonlinearity_name='concat_elu', gated=True):
  """Builds conv part of net.
  Args:
    inputs: input images
    keep_prob: dropout layer
  """
  nonlinearity = nn.set_nonlinearity(nonlinearity_name)
  filter_size = 8
  # store for as
  a = []
  # res_1
  x = inputs
  for i in xrange(nr_res_blocks):
    x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_1_" + str(i))
  # res_2
  a.append(x)
  filter_size = 2 * filter_size
  x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_2_downsample")
  for i in xrange(nr_res_blocks):
    x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_2_" + str(i))
  # res_3
  a.append(x)
  filter_size = 2 * filter_size
  x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_3_downsample")
  for i in xrange(nr_res_blocks):
    x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_3_" + str(i))
  # res_4
  a.append(x)
  filter_size = 2 * filter_size
  x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_4_downsample")
  for i in xrange(nr_res_blocks):
    x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_4_" + str(i))
  # res_4
  a.append(x)
  filter_size = 2 * filter_size
  x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_5_downsample")
  for i in xrange(nr_res_blocks):
    x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_5_" + str(i))
  # res_up_1
  filter_size = filter_size /2
  x = nn.transpose_conv_layer(x, 3, 2, filter_size, "up_conv_1")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = nn.res_block(x, a=a[-1], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_1_" + str(i))
    else:
      x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_1_" + str(i))
  # res_up_1
  filter_size = filter_size /2
  x = nn.transpose_conv_layer(x, 3, 2, filter_size, "up_conv_2")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = nn.res_block(x, a=a[-2], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_2_" + str(i))
    else:
      x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_2_" + str(i))

  filter_size = filter_size /2
  x = nn.transpose_conv_layer(x, 3, 2, filter_size, "up_conv_3")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = nn.res_block(x, a=a[-3], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_3_" + str(i))
    else:
      x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_3_" + str(i))
 
  filter_size = filter_size /2
  x = nn.transpose_conv_layer(x, 3, 2, filter_size, "up_conv_4")
  #x = PS(x,2,512)
  for i in xrange(nr_res_blocks):
    if i == 0:
      x = nn.res_block(x, a=a[-4], filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_4_" + str(i))
    else:
      x = nn.res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_4_" + str(i))
  
  x = nn.conv_layer(x, 3, 1, 3, "last_conv")
  #x = tf.nn.tanh(x) 
  #x = .9 * tf.reshape(D2Q9.WEIGHTS, [1,1,1,9]) * x
  #x = tf.reshape(D2Q9.WEIGHTS, [1,1,1,9]) + x
  pyramid_true_flow = [true_flow]
  pyramid_predicted_flow = [x]

  #tf.summary.image('sflow_p_x', x[:,:,:,1:2])
  #tf.summary.image('sflow_p_v', x[:,:,:,0:1])

  return pyramid_true_flow, pyramid_predicted_flow

def res_VAE_encoding(inputs, hidden_size=100, filter_size=8, nr_downsamples=3, nr_residual_blocks=2, gated=True, nonlinearity="concat_elu"):
  # set nonlinearity
  nonlinearity = nn.set_nonlinearity(nonlinearity)
  # encoding piece
  x_i = inputs
  for i in xrange(nr_downsamples):
    for j in xrange(nr_residual_blocks):
      x_i = nn.res_block(x_i, filter_size=filter_size, gated=gated, nonlinearity=nonlinearity, name="res_encode_" + str(i) + "_block_" + str(j))
    if i < nr_downsamples-1:
      filter_size = filter_size * 2
      x_i = nn.res_block(x_i, filter_size=filter_size, gated=gated, nonlinearity=nonlinearity, stride=2, name="res_encode_" + str(i) + "_block_" + str(nr_residual_blocks))

  # mean and stddev piece
  x_i = nn.fc_layer(x_i, hidden_size*2, "encode_layer", None, True)
  mean, stddev = tf.split(x_i, 2, 1)
  stddev =  tf.sqrt(tf.exp(stddev))
  epsilon = tf.random_normal(mean.get_shape())
  x_i_sampled = mean + epsilon * stddev

  return mean, stddev, x_i_sampled

def res_VAE_decoding(inputs, shape, filter_size=8, nr_downsamples=3, nr_residual_blocks=2, gated=True, nonlinearity="concat_elu"):
  # new shape
  new_shape = [int(inputs.get_shape()[0]), shape[0]/pow(2,nr_downsamples-1),shape[1]/pow(2,nr_downsamples-1),filter_size*pow(2,nr_downsamples-1)]
  filter_size = filter_size*pow(2,nr_downsamples-1)
  # set nonlinearity
  nonlinearity = nn.set_nonlinearity(nonlinearity)
  x_i = inputs
  x_i = nn.fc_layer(x_i, new_shape[1]*new_shape[2]*new_shape[3], "decode_layer", nn.set_nonlinearity("relu"))
  x_i = tf.reshape(x_i, new_shape)

  # decoding piece
  for i in xrange(nr_downsamples-1):
    filter_size = filter_size / 2
    x_i = nn.transpose_conv_layer(x_i, 4, 2, filter_size, "up_conv_" + str(i))
    for j in xrange(nr_residual_blocks):
      x_i = nn.res_block(x_i, filter_size=filter_size, gated=gated, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_" + str(j))
  x_i = nn.conv_layer(x_i, 3, 1, 1, "final_conv")
  x_i = tf.nn.sigmoid(x_i)

  return x_i

def res_generator(batch_size, shape, inputs=None, full_shape=None, hidden_size=100, filter_size=8, nr_downsamples=3, nr_residual_blocks=1, gated=True, nonlinearity="relu"):

  if inputs is None:
    inputs = tf.random_normal([batch_size, hidden_size])
  # new shape
  new_shape = [int(inputs.get_shape()[0]), shape[0]/pow(2,nr_downsamples-1),shape[1]/pow(2,nr_downsamples-1),filter_size*pow(2,nr_downsamples-1)]
  print(new_shape)
  filter_size = filter_size*pow(2,nr_downsamples-1)
  print(filter_size)
  # set nonlinearity
  nonlinearity = nn.set_nonlinearity(nonlinearity)
  x_i = inputs
  x_i = nn.fc_layer(x_i, new_shape[1]*new_shape[2]*new_shape[3], "decode_layer", nn.set_nonlinearity("relu"))
  x_i = tf.reshape(x_i, new_shape)

  # decoding piece
  for i in xrange(nr_downsamples-1):
    filter_size = filter_size / 2
    print(filter_size)
    x_i = nn.transpose_conv_layer(x_i, 4, 2, filter_size, "up_conv_" + str(i))
    for j in xrange(nr_residual_blocks):
      x_i = nn.res_block(x_i, filter_size=filter_size, gated=gated, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_" + str(j), normalize="batch_norm")
  x_i = nn.conv_layer(x_i, 3, 1, 1, "final_conv")
  x_i = tf.sigmoid(x_i)

  if full_shape is not None:
    x_i = tf.pad(x_i,
         [[0,0], [shape[0]/2-1,full_shape[0]-(3*shape[0]/2)-1],
                 [shape[1]/2-1,full_shape[1]-(3*shape[1]/2)-1], [0,0]])
    x_i = x_i - 1.0
    x_i = tf.pad(x_i,
         [[0,0], [1,1], [1,1], [0,0]])
    x_i = x_i + 1.0

  return x_i
res_generator_template = tf.make_template('res_generator_template', res_generator)

def res_discriminator(inputs, filter_size=8, nr_downsamples=3, nr_residual_blocks=0, gated=True, nonlinearity="relu"):
  # set nonlinearity
  nonlinearity = nn.set_nonlinearity(nonlinearity)
  # encoding piece
  x_i = inputs
  x_i = nn.conv_layer(x_i, 3, 1, filter_size, "begin_conv")
  for i in xrange(nr_downsamples):
    for j in xrange(nr_residual_blocks):
      x_i = nn.res_block(x_i, filter_size=filter_size, gated=gated, nonlinearity=nonlinearity, name="res_encode_" + str(i) + "_block_" + str(j), normalize="layer_norm")
    if i < nr_downsamples-1:
      filter_size = filter_size * 2
      x_i = nn.res_block(x_i, filter_size=filter_size, gated=gated, nonlinearity=nonlinearity, stride=2, name="res_encode_" + str(i) + "_block_" + str(nr_residual_blocks), normalize="layer_norm")

  # mean and stddev piece
  x_i = nn.fc_layer(x_i, 1, "encode_layer", None, True)
  return x_i
res_discriminator_template = tf.make_template('res_discriminator_template', res_discriminator)

def fc_conv(inputs, shape, nonlinearity_name="elu"):
  nonlinearity = nn.set_nonlinearity(nonlinearity_name)
  fc_1 = nn.fc_layer(inputs, shape[0]*shape[1]/8, 0, nonlinearity=nonlinearity)
  fc_1 = tf.reshape(fc_1, [-1, shape[0]/16, shape[0]/8, 32])
  fconv_1 = nn.transpose_conv_layer(fc_1, 3, 2, 32, "up_conv_1", nonlinearity=nonlinearity)
  fconv_2 = nn.transpose_conv_layer(fconv_1, 3, 2, 16, "up_conv_2", nonlinearity=nonlinearity)
  fconv_3 = nn.transpose_conv_layer(fconv_2, 3, 2, 8, "up_conv_3", nonlinearity=nonlinearity)
  boundary = nn.transpose_conv_layer(fconv_3, 3, 2, 1, "up_conv_4", nonlinearity=nonlinearity)
  boundary = nn.tf.sigmoid(boundary)
  return boundary
 



