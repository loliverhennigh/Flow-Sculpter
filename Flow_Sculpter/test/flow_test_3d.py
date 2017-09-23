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
from inputs.flow_data import Sailfish_data
from utils.experiment_manager import make_checkpoint_path

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS, network="flow")
print(FLOW_DIR)

shape = [96,96,96]
dims = 3
obj_size = 64
batch_size=1

def evaluate():
  """Run Eval once.
  """
  with tf.Session() as sess:
    # Make image placeholder
    boundary, true_flow = flow_net.inputs_flow(batch_size=1, shape=shape, dims=dims)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    predicted_flow = flow_net.inference_flow(boundary, 1.0)

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
    dataset = Sailfish_data("../../data/")
    dataset.load_data(dims, obj_size)
   
    #for run in filenames:
    for i in xrange(10):
      # read in boundary
      batch_boundary, batch_flow = dataset.minibatch(train=True, batch_size=1)
      #boundary_car = make_car_boundary(shape=shape, car_shape=(int(shape[1]/2.3), int(shape[0]/1.6)))

      # calc flow 
      p_flow = sess.run(predicted_flow,feed_dict={boundary: batch_boundary})
      dim=1
      print(p_flow.shape)
      print(batch_flow.shape)
      sflow_plot = np.concatenate([p_flow[...,dim], batch_flow[...,dim], np.abs(p_flow - batch_flow)[...,dim], batch_boundary[...,0]/20.0], axis=1)
      #sflow_plot = sflow_plot[0,:,:,0]
      sflow_plot = sflow_plot[0,:,:,48]
      #sflow_plot = sflow_plot[0,:,:,2]

      # display it
      plt.imshow(sflow_plot)
      plt.colorbar()
      plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
