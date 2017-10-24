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
from subprocess import call

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import model.flow_net as flow_net 
from model.pressure import calc_force
from utils.boundary_utils import wing_boundary_2d
from utils.experiment_manager import make_checkpoint_path

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS


shape = FLAGS.shape.split('x')
shape = map(int, shape)
run_steps = 30

def evaluate():
  """Run Eval once.
  """

  with tf.Graph().as_default():
    # Make image placeholder
    params_op, params_op_init, params_op_set, squeeze_loss = flow_net.inputs_boundary_learn(FLAGS.batch_size)

    # Make boundary
    boundary = flow_net.inference_boundary(FLAGS.batch_size, FLAGS.dims*[FLAGS.obj_size], params_op, full_shape=shape)

    # predict steady flow on boundary
    predicted_flow = flow_net.inference_network(boundary)

    # quantities to optimize
    force = calc_force(boundary, predicted_flow[...,2:3])
    drag_x = tf.reduce_sum(force[...,0], axis=[1,2])
    drag_y = tf.reduce_sum(force[...,1], axis=[1,2])
    drag_lift_ratio = (drag_x/drag_y)

    # loss
    loss = -tf.reduce_sum(drag_lift_ratio)

    # train_op
    variables_to_train = tf.all_variables()
    variables_to_train = [variable for i, variable in enumerate(variables_to_train) if "params" in variable.name[:variable.name.index(':')]]
    train_step = flow_net.train(loss, FLAGS.boundary_learn_lr, train_type="boundary_params", variables=variables_to_train)

    # init graph
    init = tf.global_variables_initializer()

    # start ses and init
    sess = tf.Session()
    sess.run(init)
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    params_np = (np.random.rand(FLAGS.batch_size, FLAGS.nr_boundary_params) - .5)
 
    sess.run(params_op_init, feed_dict={params_op_set: params_np})

    sess.run(train_step, feed_dict={})
    t = time.time()
    for i in tqdm(xrange(run_steps)):
      sess.run(train_step, feed_dict={})
    elapsed = time.time() - t

    filename = "./figs/learn_step_shape_" + FLAGS.shape  + "_batch_size_" + str(FLAGS.batch_size) + ".txt"
    with open(filename, "w") as f:
      f.write(str(elapsed/float(FLAGS.batch_size*run_steps)))


     

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
