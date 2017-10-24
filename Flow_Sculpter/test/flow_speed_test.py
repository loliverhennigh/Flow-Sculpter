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

from tqdm import *
#import seaborn

FLAGS = tf.app.flags.FLAGS

shape = FLAGS.shape.split('x')
shape = map(int, shape)
run_steps = 30

def evaluate():
  """Run Eval once.
  """
  with tf.Session() as sess:
    # Make image placeholder
    boundary, true_flow = flow_net.inputs_flow(batch_size=FLAGS.batch_size, shape=shape, dims=FLAGS.dims)

    # inference model.
    predicted_flow = flow_net.inference_network(boundary)

    # Restore for eval
    init = tf.global_variables_initializer()
    sess.run(init)
   
    # make graph def 
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    # make fake data
    batch_boundary = np.zeros([FLAGS.batch_size] + shape + [1])

    sess.run(predicted_flow,feed_dict={boundary: batch_boundary})
    t = time.time()
    for i in tqdm(xrange(run_steps)):
      # calc flow 
      sess.run(predicted_flow,feed_dict={boundary: batch_boundary})
    elapsed = time.time() - t

    filename = "./figs/" + FLAGS.flow_model + "_shape_" + FLAGS.shape + "_batch_size_" + str(FLAGS.batch_size) + ".txt"
    with open(filename, "w") as f:
      f.write(str(elapsed/float(FLAGS.batch_size*run_steps)))

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
