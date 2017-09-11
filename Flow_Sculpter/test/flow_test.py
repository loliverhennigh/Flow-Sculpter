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

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS, network="flow")


shape = [256, 256]
#shape = [128, 512]
dims = 2
#obj_size = 128
obj_size = 128
nr_pyramids = 0

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_key(s):
  return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def make_car_boundary(shape, car_shape):
  img = cv2.imread("../cars/car_001.png", 0)
  img = cv2.flip(img, 1)
  resized_img = cv2.resize(img, car_shape)
  resized_img = -np.rint(resized_img/255.0).astype(int).astype(np.float32) + 1.0
  resized_img = resized_img.reshape([1, car_shape[1], car_shape[0], 1])
  boundary = np.zeros((1, shape[0], shape[1], 1), dtype=np.float32)
  boundary[:, shape[0]-car_shape[1]:, 32:32+car_shape[0], :] = resized_img
  boundary[:,0,:,:] = 1.0
  boundary[:,shape[0]-1,:,:] = 1.0
  return boundary

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
    boundary, true_flow = flow_net.inputs_flow(batch_size=1, shape=shape, dims=FLAGS.dims)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    pyramid_true_flow, pyramid_predicted_flow = flow_net.inference_flow(boundary, true_flow, 1.0, nr_pyramids=nr_pyramids)

    # Restore for eval
    init = tf.global_variables_initializer()
    sess.run(init)
    variables_to_restore = tf.all_variables()
    variables_to_restore_flow = [variable for i, variable in enumerate(variables_to_restore) if "flow_network" in variable.name[:variable.name.index(':')]]
    saver = tf.train.Saver(variables_to_restore_flow)
    ckpt = tf.train.get_checkpoint_state(FLOW_DIR)
    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    # make vtm dataset
    dataset = VTK_data("../../data/")
    dataset.load_data(dims, obj_size)
   
    #for run in filenames:
    for i in xrange(10):
      # read in boundary
      batch_boundary, batch_flow = dataset.minibatch(train=True, batch_size=1)
      #boundary_car = make_car_boundary(shape=shape, car_shape=(int(shape[1]/2.3), int(shape[0]/1.6)))

      # calc flow 
      p_flow = sess.run(pyramid_predicted_flow,feed_dict={boundary: batch_boundary})[-1]
      dim=2
      sflow_plot = np.concatenate([p_flow[...,dim], batch_flow[...,dim], np.abs(p_flow - batch_flow)[...,dim], batch_boundary[...,0]/20.0], axis=1)
      #sflow_plot = sflow_plot[0,:,:,0]
      sflow_plot = sflow_plot[0,:,:]
      #sflow_plot = sflow_plot[0,:,:,2]

      # display it
      plt.imshow(sflow_plot)
      plt.colorbar()
      plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
