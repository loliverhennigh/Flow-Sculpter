#!/usr/bin/env python

import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import model.flow_net as flow_net
from inputs.heat_data_queue import Heat_Sink_data
from utils.experiment_manager import make_checkpoint_path
from model.optimizer import *

from tqdm import *
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

TRAIN_DIR = make_checkpoint_path(FLAGS.base_dir_heat, FLAGS, network="heat")

shape = FLAGS.shape.split('x')
shape = map(int, shape)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():

    # store grad and loss values
    grads = []
    loss_gen = []
    
    # global step counter
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # do for all gpus
    for i in range(FLAGS.nr_gpus):
      print("Unrolling on gpu:" + str(i))
      with tf.device('/gpu:%d' % i):
        # make inputs
        boundary, true_heat = flow_net.inputs_heat(FLAGS.batch_size, shape, FLAGS.dims) 
        # create and unrap network
        predicted_heat = flow_net.inference_network(boundary, network_type="heat", keep_prob=FLAGS.keep_prob) 
        # if i is one then get variables to store all trainable params and 
        if i == 0:
          all_params = tf.trainable_variables()
        # calc error
        error = flow_net.loss_heat(true_heat, predicted_heat)
        loss_gen.append(error)
        # store grads
        grads.append(tf.gradients(loss_gen[i], all_params))

    # exponential moving average for training
    ema = tf.train.ExponentialMovingAverage(decay=.9995)
    maintain_averages_op = tf.group(ema.apply(all_params))

    # store up the loss and gradients on gpu:0
    with tf.device('/gpu:0'):
      for i in range(1, FLAGS.nr_gpus):
        loss_gen[0] += loss_gen[i]
        for j in range(len(grads[0])):
          grads[0][j] += grads[i][j]

      # train (hopefuly)
      train_op = tf.group(adam_updates(all_params, grads[0], lr=FLAGS.lr, mom1=0.95, mom2=0.9995), maintain_averages_op, global_step.assign_add(1))

    # set total loss for printing
    total_loss = loss_gen[0]
    tf.summary.scalar('total_loss', total_loss)

    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   
    #for i, variable in enumerate(variables):
    #  print '----------------------------------------------'
    #  print variable.name[:variable.name.index(':')]

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    sess.run(init)
 
    # init from checkpoint
    variables_to_restore = tf.all_variables()
    variables_to_restore_heat = [variable for i, variable in enumerate(variables_to_restore) if ("heat_network" in variable.name[:variable.name.index(':')]) or ("global_step" in variable.name[:variable.name.index(':')])]
    saver_restore = tf.train.Saver(variables_to_restore_heat)
    ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    if ckpt is not None:
      print("init from " + TRAIN_DIR)
      try:
         saver_restore.restore(sess, ckpt.model_checkpoint_path)
      except:

         tf.gfile.DeleteRecursively(TRAIN_DIR)
         tf.gfile.MakeDirs(TRAIN_DIR)
         print("there was a problem using variables in checkpoint, random init will be used instead")

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph_def=graph_def)

    # calc number of steps left to run
    run_steps = FLAGS.max_steps - int(sess.run(global_step))

    # make vtm dataset
    dataset = Heat_Sink_data("../../data/", size=128, dim=FLAGS.dims)
    dataset.parse_data()
      
    for step in xrange(run_steps):
      current_step = sess.run(global_step)
      t = time.time()
      batch_boundary, batch_heat = dataset.minibatch(batch_size=FLAGS.batch_size, signed_distance_function=FLAGS.sdf)
      _ , loss_value = sess.run([train_op, error],feed_dict={boundary:batch_boundary, true_heat:batch_heat})
      elapsed = time.time() - t

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if current_step%100 == 0:
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))

      if current_step%1000 == 0:
        summary_str = sess.run(summary_op, feed_dict={boundary:batch_boundary, true_heat:batch_heat})
        summary_writer.add_summary(summary_str, current_step) 
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step)  
        print("saved to " + TRAIN_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
