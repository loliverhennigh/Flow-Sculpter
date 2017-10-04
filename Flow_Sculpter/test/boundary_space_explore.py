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
from model.pressure import calc_force
from utils.boundary_utils import get_random_params
from utils.experiment_manager import make_checkpoint_path

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from tqdm import *

FLAGS = tf.app.flags.FLAGS

# video init
shape = FLAGS.shape.split('x')
shape = map(int, shape)
batch_size=1

# num_frames_save
nr_frame_saves = 25

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS, network="flow")
BOUNDARY_DIR = make_checkpoint_path(FLAGS.base_dir_boundary, FLAGS, network="boundary")
print("flow dir is " + FLOW_DIR)
print("boundary dir is " + BOUNDARY_DIR)

def tile_frames(frames):

  print(frames[0].shape)
  nr_frames = len(frames)
  print(nr_frames)
  height = int(np.sqrt(nr_frames))
  print(height)
  width  = int(np.sqrt(nr_frames))
  new_frames = []
  for i in xrange(height):
    new_frames.append(np.concatenate(frames[i*height:(i+1)*height], axis=0))
    print(np.concatenate(frames[i*height:(i+1)*height], axis=0).shape)
  print(new_frames[0].shape)
  new_frames = np.concatenate(new_frames, axis=1)
  print(new_frames.shape)
  return new_frames

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
    inputs_vector, true_boundary = flow_net.inputs_boundary(FLAGS.nr_boundary_params, batch_size, shape) 

    # Build a Graph that computes the logits predictions from the
    # inference model.
    #inputs_vector_noise = inputs_vector + tf.random_normal(shape=tf.shape(inputs_vector), mean=0.0, stddev=0.0001, dtype=tf.float32) 
    boundary = flow_net.inference_boundary(1, FLAGS.dims*[FLAGS.obj_size], inputs=inputs_vector, full_shape=shape)
    #boundary = tf.round(boundary)
    predicted_flow = flow_net.inference_flow(boundary, 1.0)

    # quantities to optimize
    force = calc_force(boundary, predicted_flow[:,:,:,2:3])
    drag_x = tf.reduce_sum(force[:,:,:,0])
    drag_y = tf.reduce_sum(force[:,:,:,1])
    drag_ratio = (drag_x/drag_y)

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

    params_np = get_random_params(FLAGS.nr_boundary_params, 2)
    params_np = np.expand_dims(params_np, axis=0)
    params_np[0,0] = 0.0
    params_np[0,1] = 0.5
    params_np[0,2] = 1.0
    params_np[0,5] = 0.0

    # make store vectors for values
    resolution = 1000
    loss_val = np.zeros((resolution))
    max_d_ratio = np.zeros((resolution))
    d_ratio_store = None
    boundary_frame_store = []
    store_freq = int(resolution/nr_frame_saves)

    # make store dir
    for i in tqdm(xrange(resolution)):
      params_np[0,5] += (0.3)/resolution
      velocity_norm_g = sess.run(drag_ratio,feed_dict={inputs_vector: np.concatenate(batch_size*[params_np], axis=0)})
      if i % store_freq == 0:
        boundary_frame_store.append(sess.run(boundary,feed_dict={inputs_vector: np.concatenate(batch_size*[params_np], axis=0)})[0,int(FLAGS.obj_size/2):int(3*FLAGS.obj_size/2),int(FLAGS.obj_size/2):int(3*FLAGS.obj_size/2),0])
      loss_val[i] = velocity_norm_g

    fig = plt.figure(figsize = (10,5))
    a = fig.add_subplot(1,2,1)
    plt.title("Generated Boundary from Parameter Change")
    boundary_frame_store = tile_frames(boundary_frame_store)
    plt.imshow(boundary_frame_store)
    #plt.tick_params(axis='both', top="off", bottom="off")
    plt.axis('off')
    a = fig.add_subplot(1,2,2)
    #plt.imshow(np.concatenate(boundary_frame_store, axis = 0))
    plt.plot(np.arange(resolution)/float(resolution) - .5, loss_val)
    plt.ylabel("Loss")
    plt.xlabel("Parameter Value")
    plt.title("Loss vs Parameter Value")
    plt.savefig("./figs/boundary_space_explort.jpeg")
    plt.show() 



def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
