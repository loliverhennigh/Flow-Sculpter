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

from tvtk.api import tvtk, write_data

import sys
sys.path.append('../')

import model.flow_net as flow_net 
from inputs.flow_data import Sailfish_data
from utils.experiment_manager import make_checkpoint_path

import matplotlib.pyplot as plt

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
    
    # make graph def 
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    # make dataset
    dataset = Sailfish_data("../../data/")
    dataset.load_data(FLAGS.dims, FLAGS.obj_size)
   
    for i in xrange(10):
      # read in boundary
      batch_boundary, batch_flow = dataset.minibatch(train=True, batch_size=1)

      # calc flow 
      p_flow = sess.run(predicted_flow,feed_dict={boundary: batch_boundary})
      p_flow = p_flow * ((-batch_boundary) + 1.0)
      dim=3
      sflow_plot = np.concatenate([p_flow[...,dim], batch_flow[...,dim], np.abs(p_flow - batch_flow)[...,dim], batch_boundary[...,0]/20.0], axis=1)
      sflow_plot = sflow_plot[0,:,:,24]

      # save vtk file of it
      image_vtk = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
      image_vtk.point_data.vectors = p_flow[0,:,:,:,0:3].reshape([shape[0]*shape[1]*shape[2], 3])
      image_vtk.point_data.scalars = p_flow[0,:,:,:,3].reshape([shape[0]*shape[1]*shape[2]])
      image_vtk.point_data.scalars.name = "pressure"
      image_vtk.dimensions = shape
      write_data(image_vtk, "figs/vtk_flow_test_set_3d")

      image_vtk = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
      image_vtk.point_data.scalars = batch_boundary[0,:,:,:,0].reshape([shape[0]*shape[1]*shape[2]])
      image_vtk.point_data.scalars.name = "boundary"
      image_vtk.dimensions = shape
      write_data(image_vtk, "figs/vtk_boundary_test_set_3d")


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
