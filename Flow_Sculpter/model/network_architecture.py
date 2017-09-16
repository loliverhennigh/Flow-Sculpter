
import tensorflow as tf
import numpy as np
import nn

def res_u_network(inputs, keep_prob=1.0, filter_size=8, nr_downsamples=4, nr_residual_blocks=3, gated=True, nonlinearity="concat_elu"):
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

  x_i = nn.conv_layer(x_i, 3, 1, len(inputs.get_shape())-1, "final_conv")
  """
  # store for as
  a = []
  # set nonlinearity
  nonlinearity = nn.set_nonlinearity(nonlinearity)
  # encoding piece
  x_i = inputs
  for i in xrange(nr_downsamples):
    if i < nr_downsamples-1:
      a.append(x_i)
      filter_size = filter_size * 2
      x_i = tf.nn.avg_pool(x_i, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
      #x_i = nn.res_block(x_i, filter_size=filter_size, keep_p=keep_prob, gated=gated, nonlinearity=nonlinearity, stride=2, name="res_encode_" + str(i) + "_block_" + str(nr_residual_blocks))
  # decoding piece
  for i in xrange(nr_downsamples-1):
    filter_size = filter_size / 2
    x_i = nn.transpose_conv_layer(x_i, 4, 2, filter_size, "up_conv_" + str(i))
    x_i = nn.res_block(x_i, a=a.pop(), filter_size=filter_size, keep_p=keep_prob, gated=gated, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_0")
    for j in xrange(nr_residual_blocks-1):
      x_i = nn.res_block(x_i, filter_size=filter_size, keep_p=keep_prob, gated=gated, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_" + str(j+1))

  x_i = nn.conv_layer(x_i, 3, 1, len(inputs.get_shape())-1, "final_conv")
  """
  return x_i

# res u net template
res_u_template = tf.make_template('mini_res_u_template', res_u_network)

def res_generator_network(batch_size, shape, inputs=None, full_shape=None, hidden_size=100, filter_size=8, nr_residual_blocks=1, gated=True, nonlinearity="concat_elu"):

  # new shape
  nr_upsamples = int(np.log2(shape[0]/8))
  filter_size = filter_size*pow(2,nr_upsamples)

  # set nonlinearity
  nonlinearity = nn.set_nonlinearity(nonlinearity)

  # fc layer
  x_i = inputs
  x_i = nn.fc_layer(x_i, pow(8,len(shape))*filter_size, "decode_layer", nn.set_nonlinearity("relu"))
  x_i = tf.reshape(x_i, [batch_size] + len(shape)*[8] + [filter_size])

  # decoding piece
  for i in xrange(nr_upsamples):
    filter_size = filter_size / 2
    x_i = nn.transpose_conv_layer(x_i, 4, 2, filter_size, "up_conv_" + str(i))
    for j in xrange(nr_residual_blocks):
      x_i = nn.res_block(x_i, filter_size=filter_size, gated=gated, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_" + str(j))
  x_i = nn.conv_layer(x_i, 3, 1, 1, "final_conv")
  x_i = tf.sigmoid(x_i)

  if full_shape is not None:
    x_i = tf.pad(x_i,
         [[0,0], [shape[0]/2-2,full_shape[0]-(3*shape[0]/2)-2],
                 [shape[1]/2-2,full_shape[1]-(3*shape[1]/2)], [0,0]])
    x_i = x_i - 1.0
    x_i = tf.pad(x_i,
         [[0,0], [2,2], [2,0], [0,0]])
    x_i = x_i + 1.0

  return x_i

res_generator_template = tf.make_template('res_generator_template', res_generator_network)




