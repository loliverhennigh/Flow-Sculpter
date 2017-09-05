from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import cv2
import csv
import re
from glob import glob as glb

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import model.flow_net as flow_net 
from inputs.vtk_data import VTK_data
from utils.experiment_manager import make_checkpoint_path

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

BOUNDARY_DIR = make_checkpoint_path(FLAGS.base_dir_boundary, FLAGS, network="boundary")

shape = [64, 256]
dims = 2
obj_size = 32
nr_pyramids = 0

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_key(s):
  return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def evaluate():
  """Run Eval once.
  """
  # get a list of image filenames
  filenames = glb('../data/computed_car_flow/*')
  filenames.sort(key=alphanum_key)
  filename_len = len(filenames)
  batch_size=1
  #shape = [128, 512]
  #shape = [256, 1024]
  #shape = [128, 256]
  #shape = [64, 256]
  #shape = [16, 64]

  with tf.Session() as sess:
    # Make image placeholder
    input_dims = FLAGS.nr_boundary_params
    input_vector, true_boundary = flow_net.inputs_boundary(input_dims, batch_size=1, shape=shape)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    #mean, stddev, x_sampled, predicted_boundary = flow_net.inference_boundary(true_boundary, shape) 
    predicted_boundary = flow_net.inference_boundary_generator(1, [32,32], input_vector, full_shape=[64,256]) 

    # Restore for eval
    init = tf.global_variables_initializer()
    sess.run(init)
    variables_to_restore = tf.all_variables()
    variables_to_restore_boundary = [variable for i, variable in enumerate(variables_to_restore) if "boundary_network" in variable.name[:variable.name.index(':')]]
    saver = tf.train.Saver(variables_to_restore_boundary)
    print(BOUNDARY_DIR)
    ckpt = tf.train.get_checkpoint_state(BOUNDARY_DIR)
    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    #for run in filenames:
    for i in xrange(1000):
      # read in boundary
      #batch_boundary, batch_flow = dataset.minibatch(train=False, batch_size=1)

      # calc flow 
      input_batch, boundary_batch = flow_net.feed_dict_boundary(input_dims, 1, shape)
      p_boundary = sess.run(predicted_boundary,feed_dict={input_vector: input_batch} )
      #p_boundary = sess.run(predicted_boundary,feed_dict={true_boundary: batch_boundary})
      dim=0
      #sflow_plot = np.concatenate([p_boundary, batch_boundary], axis=1)
      sflow_plot = p_boundary
      sflow_plot = sflow_plot[0,:,:,0]
      #sflow_plot = sflow_plot[0,:,:,2]

      # display it
      plt.imshow(sflow_plot)
      plt.colorbar()
      plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
