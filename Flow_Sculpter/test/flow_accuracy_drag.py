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
from model.pressure import force_2d

import matplotlib.pyplot as plt
from tqdm import *

FLAGS = tf.app.flags.FLAGS

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS, network="flow")

shape = [128,128]
dims = 2
obj_size = 64
batch_size = 8

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
  with tf.Session() as sess:
    # Make image placeholder
    boundary, true_flow = flow_net.inputs_flow(batch_size=8, shape=shape, dims=FLAGS.dims)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    predicted_flow = flow_net.inference_flow(boundary, 1.0)

    # predict force
    predicted_force = force_2d(boundary, predicted_flow[:,:,:,2:3])
    predicted_drag_x = tf.reduce_sum(predicted_force[:,:,:,0], axis=[1,2])
    predicted_drag_y = tf.reduce_sum(predicted_force[:,:,:,1], axis=[1,2])
    true_force = force_2d(boundary, true_flow[:,:,:,2:3])
    true_drag_x = tf.reduce_sum(true_force[:,:,:,0], axis=[1,2])
    true_drag_y = tf.reduce_sum(true_force[:,:,:,1], axis=[1,2])

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
  
    # store drag data
    p_drag_x_data = []
    t_drag_x_data = []
    p_drag_y_data = []
    t_drag_y_data = []
 
    #for run in filenames:
    for i in tqdm(xrange(10)):
      # read in boundary
      batch_boundary, batch_flow = dataset.minibatch(train=False, batch_size=batch_size, signed_distance_function=FLAGS.sdf)
      #boundary_car = make_car_boundary(shape=shape, car_shape=(int(shape[1]/2.3), int(shape[0]/1.6)))

      # calc flow 
      p_drag_x, t_drag_x, p_drag_y, t_drag_y = sess.run([predicted_drag_x, true_drag_x, predicted_drag_y, true_drag_y],feed_dict={boundary: batch_boundary, true_flow: batch_flow})
      p_drag_x_data.append(p_drag_x)
      t_drag_x_data.append(t_drag_x)
      p_drag_y_data.append(p_drag_y)
      t_drag_y_data.append(t_drag_y)

    # display it
    p_drag_x_data = np.concatenate(p_drag_x_data, axis=0)
    t_drag_x_data = np.concatenate(t_drag_x_data, axis=0)
    p_drag_y_data = np.concatenate(p_drag_y_data, axis=0)
    t_drag_y_data = np.concatenate(t_drag_y_data, axis=0)
    plt.scatter(p_drag_x_data, t_drag_x_data)
    plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
