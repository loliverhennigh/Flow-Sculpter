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
shape = [256, 1024]
dims = 2
obj_size = 128
nr_pyramids = 0

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
  # get a list of image filenames
  filenames = glb('../data/computed_car_flow/*')
  filenames.sort(key=alphanum_key)
  filename_len = len(filenames)
  batch_size=1
  #set_angle = [0.035, 0.0, -0.035, -0.07]
  #set_angle = [0.035, 0.0175, 0.0, -0.0175, -0.035, -0.07, -0.15]
  set_angle = np.array([5.0, 2.5, 0.0, -2.5, -5.0, -7.5, -10.0, -12.5, -15.0, -17.5]) * (3.14159/180.0)
  desired_lift = np.array([0.0, 0.25, 0.55, 0.81, 1.05, 1.275, 1.473, 1.65, 1.7, 1.7]) * 7.5

  with tf.Graph().as_default():
    # Make image placeholder
    params_op, params_op_init, params_op_set = flow_net.inputs_boundary_learn(batch_size, set_angle=set_angle)
    boundary = flow_net.inference_boundary_generator(1, [obj_size,obj_size], inputs=params_op, full_shape=shape)
    boundary = (2.0 * boundary - 1.0)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    pyramid_true_flow, pyramid_predicted_flow = flow_net.inference_flow(boundary, boundary, 1.0, nr_pyramids=nr_pyramids)
    predicted_flow = pyramid_predicted_flow[-1]

    # quantities to optimize
    force = force_2d(boundary, predicted_flow[:,:,:,2:3])
    drag_x = tf.reduce_sum(force[:,:,:,0], axis=[0,1,2])
    drag_y = tf.reduce_sum(force[:,:,:,1], axis=[1,2])
    norm_vel = tf.sqrt(tf.square(predicted_flow[:,:,:,0:1]) + tf.square(predicted_flow[:,:,:,1:2])) * (0.5 * (-boundary + 1.0))
    ave_vel = 10.0 * tf.reduce_sum(norm_vel, axis=[1,2,3]) / tf.reduce_sum(0.5 * (-boundary + 1.0), axis=[1,2,3])
    lift_coef = (drag_y/ave_vel)

    desired_lift_tf = tf.constant(desired_lift, dtype=tf.float32)

    # loss
    if FLAGS.boundary_learn_loss == "drag_xy":
      #loss = -drag_x + tf.reduce_sum(tf.abs(desired_lift_tf - drag_y))
      loss = -drag_x/50.0 + tf.reduce_sum(tf.abs(desired_lift_tf - lift_coef))
      #loss = drag_x + drag_y
    elif FLAGS.boundary_learn_loss == "drag_x":
      loss = -drag_x
    elif FLAGS.boundary_learn_loss == "drag_y":
      loss = -drag_y

    #loss += boundary_loss

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

    #params_np = np.random.rand(batch_size,FLAGS.nr_boundary_params) * 10.0
    params_np = np.zeros((batch_size,FLAGS.nr_boundary_params-1))
 
    sess.run(params_op_init, feed_dict={params_op_set: params_np})
    run_time = FLAGS.boundary_learn_steps

    # make store vectors for values
    plot_error = np.zeros((run_time))
    plot_drag_y = np.zeros((run_time))
    plot_drag_x = np.zeros((run_time))

    # make store dir
    os.system("mkdir ./figs/boundary_learn_image_store")
    for i in tqdm(xrange(run_time)):
      l, _, d_y, d_x, l_c, p_o = sess.run([loss, train_step, drag_y, drag_x, lift_coef, params_op], feed_dict={})
      #l, _, d_y, d_x, p_o = sess.run([loss, train_step, drag_y, drag_x, bounded_params_op], feed_dict={})
      print(l_c)
      print(d_x)
      print(p_o)
      if i > 0:
        plot_error[i] = l
        plot_drag_x[i] = d_x
        plot_drag_y[i] = np.sum(d_y)
      if (i+1) % 1 == 0:
        # make video with opencv
        velocity_norm_g, boundary_g = sess.run([predicted_flow, boundary],feed_dict={})
        #velocity_norm_g, boundary_g = sess.run([force, boundary],feed_dict={})
        #sflow_plot = np.concatenate([ 5.0*velocity_norm_g[0], boundary_g[0]], axis=1)
        #sflow_plot = np.uint8(grey_to_short_rainbow(sflow_plot))
        #sflow_plot = cv2.applyColorMap(sflow_plot
        #video.write(sflow_plot)
    
        # save plot image to make video
        velocity_norm_g = velocity_norm_g[2,int(obj_size/2):int(3*obj_size/2),int(obj_size/2):int(3*obj_size/2),2]
        boundary_g = boundary_g[2,int(obj_size/2):int(3*obj_size/2),int(obj_size/2):int(3*obj_size/2),0]
        fig = plt.figure()
        fig.set_size_inches(15.5, 7.5)
        a = fig.add_subplot(1,4,1)
        plt.imshow(velocity_norm_g)
        a = fig.add_subplot(1,4,2)
        plt.imshow(boundary_g)
        a = fig.add_subplot(1,4,3)
        plt.plot(plot_error, label="loss")
        plt.plot(plot_drag_x, label="drag_x")
        plt.plot(plot_drag_y, label="drag_y")
        plt.legend()
        a = fig.add_subplot(1,4,4)
        plt.plot(set_angle, desired_lift, 'ro', label="desired lift values")
        plt.plot(set_angle, l_c, 'bo', label="true lift")
        plt.xlabel("angle of attack")
        plt.xlim(min(set_angle)-0.03, max(set_angle)+0.03)
        plt.ylim(min(desired_lift)-5.0, max(desired_lift)+5.0)
        plt.legend()
        plt.suptitle("minimizing loss " + FLAGS.boundary_learn_loss)
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
