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
from model.velocity_norm import calc_velocity_norm
from inputs.flow_data_queue import Sailfish_data
from utils.experiment_manager import make_checkpoint_path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import seaborn

FLAGS = tf.app.flags.FLAGS

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS, network="flow")

shape = FLAGS.shape.split('x')
shape = map(int, shape)

def evaluate():
  """Run Eval once.
  """
  with tf.Session() as sess:
    # Make image placeholder
    boundary, true_flow = flow_net.inputs_flow(batch_size=1, shape=shape, dims=FLAGS.dims)

    # inference model.
    predicted_flow = flow_net.inference_network(boundary, network_type="flow", keep_prob=FLAGS.keep_prob)

    # calc velocity norms
    predicted_flow_norm = calc_velocity_norm(predicted_flow)
    true_flow_norm = calc_velocity_norm(true_flow)

    # extract out pressure field
    predicted_pressure = predicted_flow[...,2]
    true_pressure = true_flow[...,2]

    # Restore for eval
    init = tf.global_variables_initializer()
    sess.run(init)
    variables_to_restore = tf.all_variables()
    variables_to_restore_flow = [variable for i, variable in enumerate(variables_to_restore) if "flow_network" in variable.name[:variable.name.index(':')]]
    saver = tf.train.Saver(variables_to_restore_flow)
    ckpt = tf.train.get_checkpoint_state(FLOW_DIR)
    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
   
    # make graph def 
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    # make dataset
    dataset = Sailfish_data("../../data/", size=FLAGS.obj_size, dim=FLAGS.dims)
    dataset.parse_data()
   
    for i in xrange(10):
      # make plot
      fig = plt.figure(figsize = (20, 10))
      gs1 = gridspec.GridSpec(2, 3)
      gs1.update(wspace=0.025, hspace=0.025)

      # read in boundary
      batch_boundary, batch_flow = dataset.minibatch(train=False, batch_size=1, signed_distance_function=FLAGS.sdf)

      # calc flow 
      p_flow_norm, p_pressure, t_flow_norm, t_pressure = sess.run([predicted_flow_norm, predicted_pressure, true_flow_norm, true_pressure],feed_dict={boundary: batch_boundary, true_flow: batch_flow})
      axarr = plt.subplot(gs1[0])
      im = axarr.imshow(p_flow_norm[0])
      axarr.get_xaxis().set_ticks([])
      axarr.get_yaxis().set_ticks([])
      axarr.set_ylabel("Velocity", y = .5, x = .5)
      axarr.set_title("Generated", y = .99)
      fig.colorbar(im, ax=axarr)
      axarr = plt.subplot(gs1[1])
      im = axarr.imshow(t_flow_norm[0])
      axarr.get_xaxis().set_ticks([])
      axarr.get_yaxis().set_ticks([])
      axarr.set_title("True", y = .99)
      fig.colorbar(im, ax=axarr)
      axarr = plt.subplot(gs1[2])
      im = axarr.imshow(np.abs(p_flow_norm[0] - t_flow_norm[0]))
      axarr.get_xaxis().set_ticks([])
      axarr.get_yaxis().set_ticks([])
      axarr.set_title("Difference", y = .99)
      fig.colorbar(im, ax=axarr)
      axarr = plt.subplot(gs1[3])
      im = axarr.imshow(p_pressure[0])
      axarr.get_xaxis().set_ticks([])
      axarr.get_yaxis().set_ticks([])
      axarr.set_ylabel("Pressure", y = .5, x = .5)
      fig.colorbar(im, ax=axarr)
      axarr = plt.subplot(gs1[4])
      im = axarr.imshow(t_pressure[0])
      axarr.get_xaxis().set_ticks([])
      axarr.get_yaxis().set_ticks([])
      fig.colorbar(im, ax=axarr)
      axarr = plt.subplot(gs1[5])
      im = axarr.imshow(np.abs(p_pressure[0] - t_pressure[0]))
      axarr.get_xaxis().set_ticks([])
      axarr.get_yaxis().set_ticks([])
      fig.colorbar(im, ax=axarr)

      plt.suptitle("Predicted vs True Steady State Flows", fontsize="x-large", y=0.94)
      plt.savefig("./figs/generated_flow_difference.jpeg")
      plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
