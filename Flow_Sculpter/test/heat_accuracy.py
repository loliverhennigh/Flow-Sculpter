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
from inputs.heat_data_queue import Heat_Sink_data
from utils.experiment_manager import make_checkpoint_path
from model.pressure import calc_force

import matplotlib.pyplot as plt
from tqdm import *

FLAGS = tf.app.flags.FLAGS

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_heat, FLAGS, network="heat")

shape = FLAGS.shape.split('x')
shape = map(int, shape)

batch_size = 10

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
    boundary, true_heat = flow_net.inputs_heat(batch_size=batch_size, shape=shape, dims=FLAGS.dims)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    predicted_heat = flow_net.inference_network(boundary, network_type="heat", keep_prob=FLAGS.keep_prob)

    # predict max heat
    predicted_max_heat = tf.reduce_max(predicted_heat, axis=[1,2,3])
    true_max_heat = tf.reduce_max(true_heat, axis=[1,2,3])

    # Restore for eval
    init = tf.global_variables_initializer()
    sess.run(init)
    variables_to_restore = tf.all_variables()
    variables_to_restore_heat = [variable for i, variable in enumerate(variables_to_restore) if "heat_network" in variable.name[:variable.name.index(':')]]
    saver = tf.train.Saver(variables_to_restore_heat)
    ckpt = tf.train.get_checkpoint_state(FLOW_DIR)
    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    # make vtm dataset
    dataset = Heat_Sink_data("../../data/", size=FLAGS.obj_size, dim=FLAGS.dims)
    dataset.parse_data()
  
    # store drag data
    p_max_heat_data = []
    t_max_heat_data = []
 
    #for run in filenames:
    for i in tqdm(xrange(60)):
      # read in boundary
      batch_boundary, batch_heat = dataset.minibatch(train=False, batch_size=batch_size)

      # calc heat 
      p_max_heat, t_max_heat = sess.run([predicted_max_heat, true_max_heat],feed_dict={boundary: batch_boundary, true_heat: batch_heat})
      p_max_heat_data.append(p_max_heat)
      t_max_heat_data.append(t_max_heat)

    # display it
    p_max_heat_data = np.concatenate(p_max_heat_data, axis=0)
    t_max_heat_data = np.concatenate(t_max_heat_data, axis=0)
    fig = plt.figure(figsize = (5,5))
    a = fig.add_subplot(1,1,1)
    plt.scatter(p_max_heat_data, t_max_heat_data)
    plt.plot(t_max_heat_data, t_max_heat_data, color="red")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Temp at Source")
    plt.savefig("./figs/heat_accuracy.pdf")
    plt.show()

    # calc average error
    print(p_max_heat_data)
    error = np.abs(p_max_heat_data - t_max_heat_data)/t_max_heat_data
    print(error)
    error = np.sum(error)/error.shape[0]
    print(error)

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
