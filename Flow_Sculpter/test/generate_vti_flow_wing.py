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
from utils.boundary_utils import get_random_params

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS, network="flow")
BOUNDARY_DIR = make_checkpoint_path(FLAGS.base_dir_boundary, FLAGS, network="boundary")

shape = FLAGS.shape.split('x')
shape = map(int, shape)

def evaluate():
  """Run Eval once.
  """
  with tf.Session() as sess:
    # Make image placeholder
    param_inputs, _ = flow_net.inputs_boundary(FLAGS.nr_boundary_params, 1, shape)

    # Make boundary
    boundary = flow_net.inference_boundary(1, FLAGS.dims*[FLAGS.obj_size], param_inputs, full_shape=shape)
    boundary = tf.round(boundary)

    # inference model.
    predicted_flow = flow_net.inference_flow(boundary, 1.0)

    # init graph
    init = tf.global_variables_initializer()
    sess.run(init)

    # Restore variables
    variables_to_restore = tf.all_variables()
    variables_to_restore_boundary = [variable for i, variable in enumerate(variables_to_restore) if "boundary_network" in variable.name[:variable.name.index(':')]]
    variables_to_restore_flow = [variable for i, variable in enumerate(variables_to_restore) if "flow_network" in variable.name[:variable.name.index(':')]]
    saver_boundary = tf.train.Saver(variables_to_restore_boundary)
    saver_flow = tf.train.Saver(variables_to_restore_flow)
    ckpt_boundary = tf.train.get_checkpoint_state(BOUNDARY_DIR)
    ckpt_flow = tf.train.get_checkpoint_state(FLOW_DIR)
    saver_boundary.restore(sess, ckpt_boundary.model_checkpoint_path)
    saver_flow.restore(sess, ckpt_flow.model_checkpoint_path)
    global_step = 1
    
    # make graph def 
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)


    for i in xrange(10):
      # random params
      rand_param = np.expand_dims(get_random_params(FLAGS.nr_boundary_params, 3), axis=0)

      # calc flow 
      p_flow, p_boundary = sess.run([predicted_flow, boundary],feed_dict={param_inputs: rand_param})
      p_flow = p_flow * ((-p_boundary) + 1.0)
      dim=3

      # plt boundary
      #plt.imshow(p_boundary[0,:,:,24,0])
      #plt.show()

      # save vtk file of it
      image_vtk = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
      image_vtk.point_data.vectors = p_flow[0,:,:,:,0:3].reshape([shape[0]*shape[1]*shape[2], 3])
      image_vtk.point_data.scalars = p_flow[0,:,:,:,3].reshape([shape[0]*shape[1]*shape[2]])
      image_vtk.point_data.scalars.name = "pressure"
      image_vtk.dimensions = shape
      write_data(image_vtk, "figs/vtk_flow_wing_3d")

      image_vtk = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
      image_vtk.point_data.scalars = p_boundary[0,:,:,:,0].reshape([shape[0]*shape[1]*shape[2]])
      image_vtk.point_data.scalars.name = "boundary"
      image_vtk.dimensions = shape
      write_data(image_vtk, "figs/vtk_boundary_wing_3d")


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
