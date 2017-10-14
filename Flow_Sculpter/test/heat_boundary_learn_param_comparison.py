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
from utils.experiment_manager import make_checkpoint_path

import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)


FLAGS = tf.app.flags.FLAGS

FLOW_DIR = make_checkpoint_path(FLAGS.base_dir_heat, FLAGS, network="heat")
BOUNDARY_DIR = make_checkpoint_path(FLAGS.base_dir_boundary_heat, FLAGS, network="boundary")

shape = FLAGS.shape.split('x')
shape = map(int, shape)
batch_size=1
num_runs = 40
std = 0.05
#temps=[.08, .02]
#temps=[0.0005, 0.001, 0.0025, 0.005, 0.01, .02]
temps=[0.00025, 0.0005, 0.001]

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
  param_new = np.minimum(np.maximum(param_new, -0.5), 0.5)
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
        l, _ = sess.run([loss, train_step], feed_dict={})
        plot_error_gradient_decent[sim, i] = np.sum(l)

    # simulated annealing
    plot_error_simulated_annealing = np.zeros((len(temps), num_runs, run_time))
    for t in tqdm(xrange(len(temps))):
      for sim in tqdm(xrange(num_runs)):
        sess.run(params_op_init, feed_dict={params_op_set: start_params_np})
        temp = temps[t]
        param_old = start_params_np 
        param_new = distort_param(start_params_np, std)
        fittness_old = sess.run(loss)
        fittness_new = 0.0
        for i in tqdm(xrange(run_time)):
          plot_error_simulated_annealing[t, sim, i] = fittness_old
          sess.run(params_op_init, feed_dict={params_op_set: param_new})
          fittness_new = sess.run(loss)
          param_old, fittness_old, temp = simulated_annealing_step(param_old, fittness_old, param_new, fittness_new, temp=temp)
          param_new = distort_param(param_old, std)

    x = np.arange(run_time)

    plot_error_gradient_decent_mean, plot_error_gradient_decent_std = calc_mean_and_std(plot_error_gradient_decent)
    plt.errorbar(x, plot_error_gradient_decent_mean, yerr=plot_error_gradient_decent_std, lw=1.0, label="Gradient Descent")

    for t in tqdm(xrange(len(temps))):
      plot_error_simulated_annealing_mean, plot_error_simulated_annealing_std = calc_mean_and_std(plot_error_simulated_annealing[t])
      #plt.errorbar(x, plot_error_simulated_annealing_mean, yerr=plot_error_simulated_annealing_std, c='g', lw=1.0, label="Simulated Annealing temp = " + str(temps[t]))
      plt.errorbar(x, plot_error_simulated_annealing_mean, yerr=plot_error_simulated_annealing_std, lw=1.0, label="Simulated Annealing temp = " + str(temps[t]))

    plt.xlabel('Step')
    plt.ylabel('Temp at Source')
    plt.title("Comparison of Gradient Descent and Simulated Annealing")
    plt.legend(loc="upper_left")
    plt.savefig("./figs/heat_learn_comparison.jpeg")
    plt.show()


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
