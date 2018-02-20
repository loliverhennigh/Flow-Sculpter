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


FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_flow, FLAGS, network="flow")
BOUNDARY_DIR = make_checkpoint_path(FLAGS.base_dir_boundary_flow, FLAGS, network="boundary")

shape = FLAGS.shape.split('x')
shape = map(int, shape)
batch_size=1
fig_pos = 2

def run_flow_solver(params, boundary, flow, sess, drag_lift_ratio_op):
  drag_lift_ratio = np.zeros(params.shape[0])
  for i in xrange(params.shape[0]):
    wing_filename = './figs/param_' + str(i)
    wing_image = wing_boundary_2d(params[i,0], params[i,1], params[i,2],
                                  params[i,3:int((FLAGS.nr_boundary_params-4)/2)],
                                  params[i,int((FLAGS.nr_boundary_params-4)/2):-1],
                                  params[i,-1], FLAGS.dims*[FLAGS.obj_size])
    wing_image = np.greater(wing_image, 0.5)
    

    print("wing image")
    print(wing_image.shape)
    #wing_image = np.rot90(wing_image, 1)
    np.save(wing_filename,wing_image)

    run_cmd = ("python steady_state_flow_" + str(FLAGS.dims) + "d.py "
             + "--vox_filename=../Flow_Sculpter/test/" + wing_filename + ".npy "
             + "--vox_size=" + str(FLAGS.obj_size) + " "
             + "--output=../Flow_Sculpter/test/figs/flow_simulation_store/param_"  + str(i) + "/step")

    call(("rm ./figs/flow_simulation_store/param_" + str(i) + "/*").split(' '), cwd="../../sailfish_flows/")
    call(run_cmd.split(' '), cwd="../../sailfish_flows/")

    wing_boundary = np.load("./figs/flow_simulation_store/param_"  + str(i) + "/step_boundary.npy")
    wing_boundary = np.expand_dims(wing_boundary, axis=0)
    wing_boundary = wing_boundary.astype(np.float32)

    wing_flow = np.load("./figs/flow_simulation_store/param_"  + str(i) + "/step_steady_flow.npy")
    wing_flow = np.expand_dims(wing_flow, axis=0)
    wing_flow = wing_flow.astype(np.float32)
    print(wing_boundary.shape)
    print(wing_flow.shape)

    #plt.imshow(wing_flow[0,:,:,2])
    #plt.show()
 
    drag_lift_ratio[i] = sess.run(drag_lift_ratio_op, feed_dict={boundary: wing_boundary, flow: wing_flow}) 
    
  return drag_lift_ratio 

def evaluate():
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """

  num_angles = 9
  max_angle =  0.30
  min_angle = -0.10
  set_params          = np.array(num_angles*[FLAGS.nr_boundary_params*[0.0]])
  set_params[:,:]     = 0.0
  set_params_pos      = np.array(num_angles*[FLAGS.nr_boundary_params*[0.0]])
  set_params_pos[:,:] = 1.0

  for i in xrange(num_angles):
    set_params[i,0]      = -i 
  set_params[:,0] = ((max_angle - min_angle) * (set_params[:,0]/(num_angles-1))) - min_angle

  set_params[:,1]      = 0.5
  set_params[:,2]      = 1.0
  set_params[:,-1]     = 0.0

  set_params_pos[:,0]  = 0.0 # set angle to 0.0
  set_params_pos[:,1]  = 0.0 # set n_1 to .5
  set_params_pos[:,2]  = 0.0 # set n_2 to 1.0
  set_params_pos[:,-1] = 0.0 # set tail hieght to 0.0

  with tf.Graph().as_default():
    # Make image placeholder
    params_op, params_op_init, params_op_set, squeeze_loss = flow_net.inputs_boundary_learn(batch_size, set_params=set_params, set_params_pos=set_params_pos, noise_std=0.01)

    # Make placeholder for flow computed by lattice boltzmann solver
    solver_boundary, solver_flow = flow_net.inputs_flow(1, shape, FLAGS.dims)
    sharp_boundary, blaa = flow_net.inputs_flow(batch_size*set_params.shape[0], shape, FLAGS.dims)

    # Make boundary
    boundary = flow_net.inference_boundary(batch_size*set_params.shape[0], FLAGS.dims*[FLAGS.obj_size], params_op, full_shape=shape)

    # predict steady flow on boundary
    predicted_flow = flow_net.inference_network(boundary, network_type="flow", keep_prob=FLAGS.keep_prob)
    sharp_predicted_flow = flow_net.inference_network(sharp_boundary, network_type="flow", keep_prob=FLAGS.keep_prob)

    # quantities to optimize
    force = calc_force(boundary, predicted_flow[...,-1:])
    sharp_force = calc_force(sharp_boundary, sharp_predicted_flow[...,-1:])
    solver_force = calc_force(solver_boundary, solver_flow[...,-1:])
    drag_x = tf.reduce_sum(force[...,0], axis=[1,2])/batch_size
    drag_y = tf.reduce_sum(force[...,1], axis=[1,2])/batch_size
    sharp_drag_x = tf.reduce_sum(sharp_force[...,0], axis=[1,2])/batch_size
    sharp_drag_y = tf.reduce_sum(sharp_force[...,1], axis=[1,2])/batch_size
    solver_drag_x = tf.reduce_sum(solver_force[...,0], axis=[1,2])/batch_size
    solver_drag_y = tf.reduce_sum(solver_force[...,1], axis=[1,2])/batch_size
    
    drag_lift_ratio        = -(drag_y/drag_x)
    sharp_drag_lift_ratio  = -(sharp_drag_y/sharp_drag_x)
    solver_drag_lift_ratio = -(solver_drag_y/solver_drag_x)

    # loss
    loss = -tf.reduce_sum(drag_lift_ratio)
    #loss = -drag_y + drag_x
    #loss = -tf.reduce_sum(drag_x)
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

    params_np = (np.random.rand(1,FLAGS.nr_boundary_params) - .5)
    #params_np = np.zeros((1,FLAGS.nr_boundary_params-1))
 
    sess.run(params_op_init, feed_dict={params_op_set: params_np})
    run_time = FLAGS.boundary_learn_steps

    # make store vectors for values
    plot_error = np.zeros((run_time))
    plot_drag_y = np.zeros((run_time))
    plot_drag_x = np.zeros((run_time))

    # make store dir
    os.system("mkdir ./figs/boundary_learn_image_store")
    for i in tqdm(xrange(run_time)):
      l, _, d_y, d_x = sess.run([loss, train_step, drag_y, drag_x], feed_dict={})
      plot_error[i] = np.sum(l)
      plot_drag_x[i] = np.sum(d_x[fig_pos])
      plot_drag_y[i] = np.sum(d_y[fig_pos])
      if ((i+1) % 1 == 0) or i == run_time-1:
        # make video with opencv
        s_params = sess.run(params_op)
        wing_boundary = []
        for p in xrange(s_params.shape[0]):
          wing_boundary.append(wing_boundary_2d(s_params[p,0], s_params[p,1], s_params[p,2],
                                           s_params[p,3:int((FLAGS.nr_boundary_params-4)/2)],
                                           s_params[p,int((FLAGS.nr_boundary_params-4)/2):-1],
                                           s_params[p,-1], FLAGS.dims*[FLAGS.obj_size]))
        wing_boundary = np.stack(wing_boundary)
        wing_boundary = np.pad(wing_boundary, [[0,0],[128,128],[128,128],[0,0]], 'constant', constant_values=0.0)
        #print(sharp_boundary.get_shape())
        #print(wing_boundary.shape)
        p_flow, p_boundary, d_l_ratio, sharp_d_l_ratio = sess.run([sharp_predicted_flow, boundary, drag_lift_ratio, sharp_drag_lift_ratio],feed_dict={sharp_boundary: wing_boundary})
    
        # save plot image to make video
        p_pressure = p_flow[fig_pos,:,:,2]
        p_boundary = p_boundary[fig_pos,:,:,0]
        fig = plt.figure()
        fig.set_size_inches(15, 10)
        a = fig.add_subplot(2,3,1)
        plt.imshow(p_pressure)
        plt.title("Pressure", fontsize=16)
        a = fig.add_subplot(2,3,2)
        plt.imshow(p_boundary)
        plt.title("Boundary", fontsize=16)
        a = fig.add_subplot(2,3,3)
        plt.plot(plot_error, label="Sum(Lift/Drag)")
        plt.xlabel("Step")
        plt.legend()
        a = fig.add_subplot(2,3,4)
        plt.plot(-plot_drag_x, label="Drag Angle 0")
        plt.plot(plot_drag_y, label="Lift Angle 0")
        plt.ylim(-1.0, np.max(plot_drag_y)+2.0)
        plt.xlabel("Step")
        plt.legend()
        a = fig.add_subplot(2,3,5)
        plt.plot(-np.degrees(set_params[:,0]), d_l_ratio, 'bo', label="Lift/Drag Network")
        #plt.plot(-np.degrees(set_params[:,0]), sharp_d_l_ratio, 'ro', label="Lift/Drag Sharp")
        #if i == run_time-1:
        #  solver_d_l_ratio = run_flow_solver(sess.run(params_op), solver_boundary, solver_flow, sess, solver_drag_lift_ratio)
        #  plt.plot(-np.degrees(set_params[:,0]), solver_d_l_ratio, 'go', label="Lift/Drag Solver")
        plt.xlabel("Angle of Attack (Degrees)")
        plt.xlim(min(-np.degrees(set_params[:,0]))-3, max(-np.degrees(set_params[:,0]))+3)
        plt.ylim(np.min(d_l_ratio)-1, np.max(d_l_ratio)+2)
        plt.legend()
        plt.suptitle("2D Wing Optimization Using Gradient Descent", fontsize=20)
        plt.savefig("./figs/boundary_learn_image_store/plot_" + str(i).zfill(5) + ".png")
        if run_time - i <= 100:
          plt.savefig("./figs/" + FLAGS.boundary_learn_loss + "_plot.pdf")
        if i == run_time - 1:
          plt.savefig("./figs/learn_gradient_descent.pdf")
          plt.show()
        #plt.show()
        plt.close(fig)


    # generate video of plots
    os.system("rm ./figs/airfoil_2d_video.mp4")
    os.system("cat ./figs/boundary_learn_image_store/*.png | ffmpeg -f image2pipe -r 30 -vcodec png -i - -vcodec libx264 ./figs/airfoil_2d_video.mp4")
    os.system("rm -r ./figs/boundary_learn_image_store")

     

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
