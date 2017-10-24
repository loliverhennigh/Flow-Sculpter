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
from utils.boundary_utils import heat_sink_boundary_2d
from utils.experiment_manager import make_checkpoint_path

import matplotlib
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_heat, FLAGS, network="heat")
BOUNDARY_DIR = make_checkpoint_path(FLAGS.base_dir_boundary_heat, FLAGS, network="boundary")

shape = FLAGS.shape.split('x')
shape = map(int, shape)
batch_size=1
num_runs = 1
std = 0.05
#temps=[.08, .02]
#temps=[0.0005, 0.001, 0.0025, 0.005, 0.01, .02]
temps=[0.0025]

def run_heat_sink_simulation(params):
  params = params + 0.5 # get back to same range
  
  boundary = heat_sink_boundary_2d(params[0], [128,128])
  print(boundary.shape)
  u = diffusion_simulation(boundary[...,0])
  return np.max(u)

def diffusion_simulation(boundary):
  # plate size, mm
  w = h = boundary.shape[0]/10.0
  # intervals in x-, y- directions, mm
  dx = 0.1
  # Thermal diffusivity of steel, mm2.s-1
  D_copper   = 400.0
  D_aluminum = 400.0
  D_air      = 1.0

  nx, ny = int(w/dx), int(h/dx)

  dx2, dx2 = dx*dx, dx*dx
  dt = dx2 * dx2 / (2 * D_copper * (dx2 + dx2))

  u0 = np.ones((nx, ny))
  u = np.empty((nx, ny))
 
  # make diffusion coef matrix 
  diffusion_coef = (D_aluminum - D_air) * (-boundary + 1.0) + D_air

  # place copper pipes
  pos_1 = (4*boundary.shape[0]/8, 15*boundary.shape[1]/16)
  #pos_2 = (5*boundary.shape[0]/8, 15*boundary.shape[1]/16)
  radius = boundary.shape[0]/24
  #cv2.circle(diffusion_coef, pos_1, radius, (D_copper), -1)
  #cv2.circle(diffusion_coef, pos_2, radius, (D_copper), -1)

  ave_px_diffusion_coef = np.minimum(diffusion_coef[1:-1,1:-1], diffusion_coef[2:,  1:-1])
  ave_nx_diffusion_coef = np.minimum(diffusion_coef[1:-1,1:-1], diffusion_coef[ :-2,1:-1])
  ave_py_diffusion_coef = np.minimum(diffusion_coef[1:-1,1:-1], diffusion_coef[1:-1,2:  ])
  ave_ny_diffusion_coef = np.minimum(diffusion_coef[1:-1,1:-1], diffusion_coef[1:-1, :-2])
  ave_all_diffusion_coef = (ave_px_diffusion_coef
                           +ave_nx_diffusion_coef
                           +ave_py_diffusion_coef
                           +ave_ny_diffusion_coef)/4.0

  def do_timestep(u0, u):
      # Propagate with forward-difference in time, central-difference in space
   
      u[1:-1, 1:-1] = u0[1:-1, 1:-1] +  dt * (
             ave_px_diffusion_coef * u0[2:,  1:-1] 
           + ave_nx_diffusion_coef * u0[ :-2,1:-1]
           + ave_py_diffusion_coef * u0[1:-1,2:  ] 
           + ave_ny_diffusion_coef * u0[1:-1, :-2]
           - 4 * ave_all_diffusion_coef * u0[1:-1, 1:-1])/dx2
      u0 = u.copy()
      return u0, u

  def apply_boundary(u0):

    u0[boundary.shape[0]-2, int(7*boundary.shape[1]/16):int(9*boundary.shape[1]/16)] += 1000.0*dt

    u0[boundary.shape[0]-1, int(1*boundary.shape[1]/8):int(7*boundary.shape[1]/8+1)] = u0[boundary.shape[0]-2, int(1*boundary.shape[1]/8):int(7*boundary.shape[1]/8+1)]

    u0[np.greater(boundary, 0.5)] = 0.0


  # Number of timesteps
  t = time.time()
  temp_prev = np.max(u0)
  while True:
    u0, u = do_timestep(u0, u)
    apply_boundary(u0)
    if np.abs(np.max(u0) - temp_prev) < .000000001:
      break
    temp_prev = np.max(u0)
  #plt.imshow(u0)
  #plt.show()
  return u0


# 2d or not
def calc_mean_and_std(values):
    values_mean = np.sum(values, axis=0) / values.shape[0]
    values_std = np.sqrt(np.sum(np.square(values - np.expand_dims(values_mean, axis=0)), axis=0)/values.shape[0])
    return values_mean, values_std

def simulated_annealing_step(param_old, fittness_old, param_new, fittness_new, temp=1.0, min_temp=0.00001, alpha=0.995):
  ap = np.exp((fittness_old - fittness_new)/temp)
  if np.random.rand() < ap:
    param = param_new
    fittness = fittness_new
  else:
    param = param_old
    fittness = fittness_old
  temp = temp * alpha 
  return param, fittness, temp

def distort_param(param, std):
  param_new = param + np.random.normal(loc=0.0, scale=std, size=param.shape)
  param_new = np.minimum(np.maximum(param_new, -0.45), 0.45)
  return param_new

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
    predicted_heat = flow_net.inference_network(boundary, network_type="heat")

    # loss
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
   
    # make graph 
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    # total run time
    run_time = FLAGS.boundary_learn_steps

    # use same start for comparison
    start_params_np = (np.random.rand(batch_size,FLAGS.nr_boundary_params) - .5)/2.0

    # gradient decent
    plot_error_gradient_decent = np.zeros((num_runs, run_time))
    for sim in tqdm(xrange(num_runs)):
      sess.run(params_op_init, feed_dict={params_op_set: start_params_np})
      for i in tqdm(xrange(run_time)):
        plot_error_gradient_decent[sim, i] = run_heat_sink_simulation(sess.run(params_op) - 0.5)
        sess.run(train_step, feed_dict={})
    gradient_descent_boundary = heat_sink_boundary_2d(sess.run(params_op)[0], [128,128])

    # simulated annealing
    plot_error_simulated_annealing = np.zeros((len(temps), num_runs, run_time))
    for t in tqdm(xrange(len(temps))):
      for sim in tqdm(xrange(num_runs)):
        temp = temps[t]
        param_old = start_params_np 
        fittness_old = run_heat_sink_simulation(param_old)
        param_new = distort_param(start_params_np, std)
        fittness_new = 0.0
        for i in tqdm(xrange(run_time)):
          plot_error_simulated_annealing[t, sim, i] = fittness_old
          fittness_new = run_heat_sink_simulation(param_new)
          param_old, fittness_old, temp = simulated_annealing_step(param_old, fittness_old, param_new, fittness_new, temp=temp)
          param_new = distort_param(param_old, std)

    simulated_annealing_boundary = heat_sink_boundary_2d(param_old[0] + .5, [128,128])

    x = np.arange(run_time)

    plot_error_gradient_decent_mean, plot_error_gradient_decent_std = calc_mean_and_std(plot_error_gradient_decent)



    fig = plt.figure()
    fig.set_size_inches(10, 5)
    a = fig.add_subplot(1,2,1)
    plt.imshow((simulated_annealing_boundary - .5*gradient_descent_boundary)[:,:,0])
    plt.title("Difference in Heat Sink Design", fontsize=16)
    a = fig.add_subplot(1,2,2)
    plt.errorbar(x, plot_error_gradient_decent_mean, yerr=plot_error_gradient_decent_std, lw=1.0, label="Gradient Descent")
    for t in tqdm(xrange(len(temps))):
      plot_error_simulated_annealing_mean, plot_error_simulated_annealing_std = calc_mean_and_std(plot_error_simulated_annealing[t])
      plt.errorbar(x, plot_error_simulated_annealing_mean, yerr=plot_error_simulated_annealing_std, lw=1.0, label="Simulated Annealing temp = " + str(temps[t]))
    plt.xlabel('Step')
    plt.ylabel('Temp at Source')
    plt.suptitle("Gradient Descent vs Simulated Annealing", fontsize=20)
    plt.legend(loc="upper_left")
    plt.savefig("./figs/heat_learn_comparison.pdf")
    plt.show()


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
