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
from tqdm import *
import os

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import model.flow_net as flow_net 
from model.pressure import force_2d
#from utils.flow_reader import load_flow, load_boundary, load_state
from utils.experiment_manager import make_checkpoint_path

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS

# video init
shape = [256, 256]
dims = 2
obj_size = 128
nr_pyramids = 0
batch_size=1

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

success = video.open('figs/' + FLAGS.boundary_learn_loss + '_video.mov', fourcc, 10, (2*shape[1], shape[0]), True)


FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS, network="flow")
BOUNDARY_DIR = make_checkpoint_path(FLAGS.base_dir_boundary, FLAGS, network="boundary")
print("flow dir is " + FLOW_DIR)
print("boundary dir is " + BOUNDARY_DIR)

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_key(s):
  return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def evaluate():
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Graph().as_default():
    # Make image placeholder
    inputs_vector, true_boundary = flow_net.inputs_boundary(12, batch_size, shape) 

    # Build a Graph that computes the logits predictions from the
    # inference model.
    inputs_vector_noise = inputs_vector + tf.random_normal(shape=tf.shape(inputs_vector), mean=0.0, stddev=0.0001, dtype=tf.float32) 
    boundary = flow_net.inference_boundary_generator(1, [obj_size,obj_size], inputs=inputs_vector_noise, full_shape=shape)
    boundary = tf.round(boundary)
    boundary = (2.0 * boundary - 1.0)
    _, pyramid_predicted_flow = flow_net.inference_flow(boundary, boundary, 1.0, nr_pyramids=nr_pyramids)
    predicted_flow = pyramid_predicted_flow[-1]

    # quantities to optimize
    force = force_2d(boundary, predicted_flow[:,:,:,2:3])
    drag_x = tf.reduce_sum(force[:,:,:,0], axis=[0,1,2])
    drag_y = tf.reduce_sum(force[:,:,:,1], axis=[0,1,2])
    drag_ratio = (drag_y/(-drag_x))

    # init graph
    init = tf.global_variables_initializer()

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = tf.all_variables()
    variables_to_restore_boundary = [variable for i, variable in enumerate(variables_to_restore) if "boundary_network" in variable.name[:variable.name.index(':')]]
    variables_to_restore_flow = [variable for i, variable in enumerate(variables_to_restore) if "flow_network" in variable.name[:variable.name.index(':')]]
    saver_boundary = tf.train.Saver(variables_to_restore_boundary)
    saver_flow = tf.train.Saver(variables_to_restore_flow)

    # start ses and init
    sess = tf.Session()
    sess.run(init)
    ckpt_boundary = tf.train.get_checkpoint_state(BOUNDARY_DIR)
    ckpt_flow = tf.train.get_checkpoint_state(FLOW_DIR)
    saver_boundary.restore(sess, ckpt_boundary.model_checkpoint_path)
    saver_flow.restore(sess, ckpt_flow.model_checkpoint_path)
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    params_np = np.array([[0.0,0.5,1.0,.0,.16,.16,.17,.16,.16,.16,.16,0.0]])

    # make store vectors for values
    resolution = 200
    loss_val = np.zeros((resolution))
    max_d_ratio = np.zeros((resolution))
    d_ratio_store = None

    # make store dir
    for i in xrange(resolution):
      params_np[0,3] += (1.0/3.0)/resolution
      velocity_norm_g = sess.run(drag_ratio,feed_dict={inputs_vector: np.concatenate(batch_size*[params_np], axis=0)})
      loss_val[i] = velocity_norm_g
    plt.plot(loss_val)
    plt.show() 



def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
