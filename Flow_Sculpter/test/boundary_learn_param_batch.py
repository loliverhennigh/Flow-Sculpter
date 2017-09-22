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

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

# video init
shape = [256, 256]
dims = 2
obj_size = 128
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
  set_params     = np.array([FLAGS.nr_boundary_params*[0.0]])
  set_params_pos = np.array([FLAGS.nr_boundary_params*[0.0]]) + 1.0
  set_params[0,1]      = 0.5
  set_params[0,2]      = 1.0
  set_params_pos[0,0]  = 0.0 # set angle to 0.0
  set_params_pos[0,1]  = 0.0 # set n_1 to .5
  set_params_pos[0,2]  = 0.0 # set n_2 to 1.0
  set_params_pos[0,-1] = 0.0 # set tail hieght to 0.0

  with tf.Graph().as_default():
    # Make image placeholder
    params_op, params_op_init, params_op_set, squeeze_loss = flow_net.inputs_boundary_learn(batch_size, set_params=set_params, set_params_pos=set_params_pos, noise_std=0.001)

    # Make boundary
    boundary = flow_net.inference_boundary(batch_size, [obj_size,obj_size], params_op, full_shape=shape)
    sharp_boundary = tf.round(boundary)
    boundary = (2.0 * boundary - 1.0)
    sharp_boundary = (2.0 * sharp_boundary - 1.0)

    # predict steady flow on boundary
    predicted_flow = flow_net.inference_flow(boundary, 1.0)
    predicted_sharp_flow = flow_net.inference_flow(boundary, 1.0)

    # quantities to optimize
    force = force_2d(boundary, predicted_flow[:,:,:,2:3])
    sharp_force = force_2d(sharp_boundary, predicted_sharp_flow[:,:,:,2:3])
    drag_x = tf.reduce_sum(force[:,:,:,0], axis=[0,1,2])/batch_size
    drag_y = tf.reduce_sum(force[:,:,:,1], axis=[0,1,2])/batch_size
    sharp_drag_x = tf.reduce_sum(sharp_force[:,:,:,0], axis=[1,2])/batch_size
    sharp_drag_y = tf.reduce_sum(sharp_force[:,:,:,1], axis=[1,2])/batch_size
    lift_coef = (drag_y/(-drag_x))
    sharp_lift_coef = (sharp_drag_y/(-sharp_drag_x))

    # loss
    loss = -tf.reduce_sum(drag_y/(-drag_x))
    loss += squeeze_loss

    # train_op
    variables_to_train = tf.all_variables()
    variables_to_train = [variable for i, variable in enumerate(variables_to_train) if "params" in variable.name[:variable.name.index(':')]]
    train_step = flow_net.train(loss, FLAGS.boundary_learn_lr, train_type="boundary_params", variables=variables_to_train)

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

    params_np = (np.random.rand(batch_size,FLAGS.nr_boundary_params) - .5)
    #params_np = np.zeros((1,FLAGS.nr_boundary_params-1))
 
    sess.run(params_op_init, feed_dict={params_op_set: params_np})
    run_time = FLAGS.boundary_learn_steps

    # make store vectors for values
    best_boundary = None
    best_flow = None
    max_d_ratio = np.zeros((run_time))
    iteration = np.arange(run_time) * batch_size
    d_ratio_store = None

    # make store dir
    os.system("mkdir ./figs/boundary_learn_image_store")
    for i in tqdm(xrange(run_time)):
      d_ratio, _, boundary_batch, velocity_batch = sess.run([sharp_lift_coef, train_step, sharp_boundary, predicted_flow], feed_dict={})
      if best_boundary is None:
        best_boundary = boundary_batch[np.argmax(d_ratio)]
        best_flow = velocity_batch[np.argmax(d_ratio)]
      elif np.max(d_ratio_store) < np.max(d_ratio):
        best_boundary = boundary_batch[np.argmax(d_ratio)]
        best_flow = velocity_batch[np.argmax(d_ratio)]
      if d_ratio_store is None:
        d_ratio_store = d_ratio
      else:
        d_ratio_store = np.concatenate([d_ratio_store, d_ratio], axis=0)
      max_d_ratio[i] = np.max(d_ratio_store)
      if i % 10 == 0:
        # make video with opencv
    
        # save plot image to make video
        velocity_norm_g = best_flow[:,:,2]
        boundary_g = best_boundary[:,:,0]
        fig = plt.figure()
        fig.set_size_inches(25.5, 7.5)
        a = fig.add_subplot(1,4,1)
        plt.imshow(velocity_norm_g)
        a = fig.add_subplot(1,4,2)
        plt.imshow(boundary_g)
        a = fig.add_subplot(1,4,3)
        plt.plot(iteration, max_d_ratio, label="best lift/drag")
        plt.legend(loc=4)
        a = fig.add_subplot(1,4,4)
        # the histogram of the data
        n, bins, patches = plt.hist(d_ratio_store, 50, normed=1, facecolor='green')
        #plt.hist(d_ratio_store, 10, normed=1, facecolor='green')
        plt.xlabel("lift/drag")
        plt.ylabel("frequency")
        plt.legend()
        plt.suptitle("Using Gradient Decent")
        plt.savefig("./figs/boundary_learn_image_store/plot_" + str(i).zfill(5) + ".png")
        if run_time - i <= 100:
          plt.savefig("./figs/" + FLAGS.boundary_learn_loss + "_plot.png")
          #plt.show()
        plt.close(fig)

    # close cv video
    video.release()
    cv2.destroyAllWindows()

    # generate video of plots
    os.system("rm ./figs/" + FLAGS.boundary_learn_loss + "_plot_video.mp4")
    os.system("cat ./figs/boundary_learn_image_store/*.png | ffmpeg -f image2pipe -r 30 -vcodec png -i - -vcodec libx264 ./figs/" + FLAGS.boundary_learn_loss + "_plot_video.mp4")
    os.system("rm -r ./figs/boundary_learn_image_store")

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
