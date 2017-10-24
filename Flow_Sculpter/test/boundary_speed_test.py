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
from inputs.vtk_data import VTK_data
from utils.experiment_manager import make_checkpoint_path

from tqdm import *

FLAGS = tf.app.flags.FLAGS


# video init
shape = FLAGS.shape.split('x')
shape = map(int, shape)
run_steps = 30

def evaluate():
  """Run Eval once.
  """
  with tf.Session() as sess:
    # Make image placeholder
    input_dims = FLAGS.nr_boundary_params
    input_vector, true_boundary = flow_net.inputs_boundary(input_dims, batch_size=FLAGS.batch_size, shape=shape)

    # Build a Graph that computes the logits predictions from the
    predicted_boundary = flow_net.inference_boundary(FLAGS.batch_size, FLAGS.dims*[FLAGS.obj_size], input_vector, full_shape=shape) 

    # Restore for eval
    init = tf.global_variables_initializer()
    sess.run(init)
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    # make one input and run on it again and again
    #input_batch, boundary_batch = flow_net.feed_dict_boundary(input_dims, FLAGS.batch_size, shape)

    sess.run(predicted_boundary,feed_dict={input_vector: np.zeros((FLAGS.batch_size, input_dims))} )
    t = time.time()
    for i in tqdm(xrange(run_steps)):
      sess.run(predicted_boundary,feed_dict={input_vector: np.zeros((FLAGS.batch_size, input_dims))} )
    elapsed = time.time() - t

    filename = "./figs/boundary_network_shape_" + FLAGS.shape + "_batch_size_" + str(FLAGS.batch_size) + ".txt"
    with open(filename, "w") as f:
      f.write(str(elapsed/float(FLAGS.batch_size*run_steps)))



def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
