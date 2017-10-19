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



#SMALL_SIZE = 12
#MEDIUM_SIZE = 16
#BIGGER_SIZE = 18

#plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

FLAGS = tf.app.flags.FLAGS

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_heat, FLAGS, network="heat")
BOUNDARY_DIR = make_checkpoint_path(FLAGS.base_dir_boundary_heat, FLAGS, network="boundary")

shape = FLAGS.shape.split('x')
shape = map(int, shape)
batch_size=1

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
    params_op, params_op_init, params_op_set, squeeze_loss = flow_net.inputs_boundary_learn(batch_size, network_type="heat", noise_std=0.01)

    # Make boundary
    boundary = flow_net.inference_boundary(batch_size, 2*[128], params_op)

    # predict steady flow on boundary
    predicted_heat = flow_net.inference_network(boundary, network_type="heat", keep_prob=1.0)

    # loss
    #loss = tf.reduce_sum(predicted_heat)
    loss = tf.reduce_max(predicted_heat)
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
    variables_to_restore_heat = [variable for i, variable in enumerate(variables_to_restore) if "heat_network" in variable.name[:variable.name.index(':')]]
    saver_boundary = tf.train.Saver(variables_to_restore_boundary)
    saver_heat = tf.train.Saver(variables_to_restore_heat)

    # start ses and init
    sess = tf.Session()
    sess.run(init)
    ckpt_boundary = tf.train.get_checkpoint_state(BOUNDARY_DIR)
    ckpt_heat = tf.train.get_checkpoint_state(FLOW_DIR)
    saver_boundary.restore(sess, ckpt_boundary.model_checkpoint_path)
    saver_heat.restore(sess, ckpt_heat.model_checkpoint_path)
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    params_np = (np.random.rand(1,FLAGS.nr_boundary_params) - .5)/2.0
 
    sess.run(params_op_init, feed_dict={params_op_set: params_np})
    run_time = FLAGS.boundary_learn_steps

    # make store vectors for values
    plot_error = np.zeros((run_time))

    # make store dir
    os.system("mkdir ./figs/boundary_learn_image_store")
    for i in tqdm(xrange(run_time)):
      l, _, = sess.run([loss, train_step], feed_dict={})
      print(l)
      plot_error[i] = np.sum(l)
      if ((i+1) % 100 == 0) or i == run_time-1:
        # make video with opencv
        p_heat, p_boundary = sess.run([predicted_heat, boundary])
    
        # save plot image to make video
        fig = plt.figure()
        fig.set_size_inches(15, 5)
        a = fig.add_subplot(1,3,2)
        plt.imshow(p_heat[0,:,:,0])
        #plt.title("Heat Dissipation", fontsize="x-large")
        plt.title("Heat Dissipation", fontsize=16)
        a = fig.add_subplot(1,3,3)
        plt.imshow(p_boundary[0,:,:,0])
        plt.title("Heat Sink Geometry", fontsize=16)
        a = fig.add_subplot(1,3,1)
        plt.plot(plot_error, label="Temp at Source")
        plt.xlabel("Step")
        plt.ylabel("Temp")
        plt.legend()
        plt.suptitle("Heat Sink Optimization Using Gradient Decent", fontsize=20)
        plt.savefig("./figs/boundary_learn_image_store/plot_" + str(i).zfill(5) + ".png")
        if run_time - i <= 100:
          plt.savefig("./figs/" + FLAGS.boundary_learn_loss + "_plot.png")
        if i == run_time - 1:
          plt.savefig("./figs/heat_learn_gradient_decent.jpeg")
          plt.show()
        plt.close(fig)


    # generate video of plots
    os.system("rm ./figs/" + FLAGS.boundary_learn_loss + "_plot_video.mp4")
    os.system("cat ./figs/boundary_learn_image_store/*.png | ffmpeg -f image2pipe -r 30 -vcodec png -i - -vcodec libx264 ./figs/" + FLAGS.boundary_learn_loss + "_plot_video.mp4")
    os.system("rm -r ./figs/boundary_learn_image_store")

     

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
