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
from model.pressure import force_2d
from inputs.flow_data import Sailfish_data
from utils.experiment_manager import make_checkpoint_path

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS, network="flow")


shape = [128,128]
dims = 2
obj_size = 64

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
    boundary, true_flow = flow_net.inputs_flow(batch_size=1, shape=shape, dims=FLAGS.dims)
    boundary_trainable = tf.Variable(np.zeros([1] + shape + [1]).astype(dtype=np.float32), name="boundary_trainable")
    boundary_trainable_init = tf.group(boundary_trainable.assign(boundary))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    predicted_flow = flow_net.inference_flow(boundary_trainable, 1.0)

    # Restore for eval
    init = tf.global_variables_initializer()
    sess.run(init)
    variables_to_restore = tf.all_variables()
    variables_to_restore_flow = [variable for i, variable in enumerate(variables_to_restore) if "flow_network" in variable.name[:variable.name.index(':')]]
    saver = tf.train.Saver(variables_to_restore_flow)
    ckpt = tf.train.get_checkpoint_state(FLOW_DIR)
    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1

    # quantities to optimize
    force = force_2d(boundary_trainable, predicted_flow[:,:,:,2:3])
    drag_x = tf.reduce_sum(force[:,:,:,0], axis=[0,1,2])
    drag_y = tf.reduce_sum(force[:,:,:,1], axis=[0,1,2])
    lift_drag_ratio = drag_x/drag_y
    #lift_drag_ratio = -drag_y
    #lift_drag_ratio = -tf.reduce_max(predicted_flow[:,:,:,0])

    # get gradient
    all_variables = tf.all_variables()
    boundary_trainable_variable = [variable for i, variable in enumerate(all_variables) if "boundary_train" in variable.name[:variable.name.index(':')]][0]
    grads = tf.gradients(lift_drag_ratio, boundary_trainable_variable)[0]
    #grads = tf.minimum(grads * ((-boundary_trainable)+1.0), 0.0) + tf.maximum(grads * boundary_trainable, 0.0)
    #grads = grads + boundary_trainable/10.0
 
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    # make vtm dataset
    dataset = Sailfish_data("../../data/")
    dataset.load_data(dims, obj_size)
   
    #for run in filenames:
    for i in xrange(10):
      # read in boundary
      batch_boundary, batch_flow = dataset.minibatch(train=True, batch_size=1, signed_distance_function=FLAGS.sdf)

      # calc flow 
      sess.run(boundary_trainable_init,feed_dict={boundary: batch_boundary})
      gradient_surface, pressure_field = sess.run([grads, predicted_flow])
      gradient_surface = gradient_surface[0,:,:,0]/4.0
      pressure_field = pressure_field[0,:,:,2]
      img = np.concatenate([gradient_surface, pressure_field], axis=0)
      #img = gradient_surface[0,:,:,0]

      # display it
      plt.imshow(img)
      plt.colorbar()
      plt.show()

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
