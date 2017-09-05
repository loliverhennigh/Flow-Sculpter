
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import model.flow_net as flow_net
from inputs.vtk_data import VTK_data
from utils.experiment_manager import make_checkpoint_path

FLAGS = tf.app.flags.FLAGS

TRAIN_DIR = make_checkpoint_path(FLAGS.base_dir_boundary, FLAGS, network="boundary")

shape = FLAGS.shape.split('x')
shape = map(int, shape)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # global step counter
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # make real boundary placeholders
    real_boundary = flow_net.inputs_boundary(0, FLAGS.batch_size, [FLAGS.obj_size, FLAGS.obj_size])

    # make generated boundary
    gen_boundary = flow_net.inference_boundary_generator(FLAGS.batch_size, [FLAGS.obj_size, FLAGS.obj_size])

    # discriminate
    error_d_real = flow_net.inference_boundary_discriminator(real_boundary)
    error_d_fake = flow_net.inference_boundary_discriminator(gen_boundary)

    # loss
    error_d = -tf.reduce_mean(error_d_real) + tf.reduce_mean(error_d_fake)
    error_g = -tf.reduce_mean(error_d_fake)
    error_g += -tf.nn.l2_loss(gen_boundary)

    # gradient penalty
    epsilon = tf.random_uniform([FLAGS.batch_size, 1,1,1], 0.0, 1.0)
    x_hat = real_boundary*epsilon + (1.0-epsilon)*gen_boundary
    d_hat = flow_net.inference_boundary_discriminator(x_hat)
    gradients = tf.gradients(d_hat, x_hat)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = 10.0*tf.reduce_mean((slopes-1.0)**2)
    error_d += gradient_penalty

    # summary
    tf.summary.scalar('d_loss', error_d)
    tf.summary.scalar('g_loss', error_g)
    tf.summary.image('g_boundary', gen_boundary)
    tf.summary.image('r_boundary', real_boundary)

    # make train op
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = [var for var in t_vars if 'generator' in var.name]
    for i, variable in enumerate(d_vars):
      print 'd---------------------------------------------'
      print variable.name[:variable.name.index(':')]
    for i, variable in enumerate(g_vars):
      print 'g---------------------------------------------'
      print variable.name[:variable.name.index(':')]

    # optimize G
    g_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.0,beta2=0.9).minimize(error_g, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)

    # optimize D
    d_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.0,beta2=0.9).minimize(error_d, var_list=d_vars, colocate_gradients_with_ops=True)

    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   

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
    variables_to_restore_flow = [variable for i, variable in enumerate(variables_to_restore) if ("boundary_network" in variable.name[:variable.name.index(':')]) or ("global_step" in variable.name[:variable.name.index(':')])]
    saver_restore = tf.train.Saver(variables_to_restore_flow)
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
    dataset = VTK_data("../../data/")
    dataset.load_data(FLAGS.dims, FLAGS.obj_size)
     
    run_steps = FLAGS.max_steps - int(sess.run(global_step))
    for step in xrange(run_steps):
      current_step = sess.run(global_step)
      t = time.time()
      for _ in xrange(5):
        batch_boundary, _ = dataset.minibatch(batch_size=FLAGS.batch_size)
        _ , loss_value_d = sess.run([d_train_op, error_d],feed_dict={real_boundary: batch_boundary})

      batch_boundary, _ = dataset.minibatch(batch_size=FLAGS.batch_size)
      _ , loss_value_g = sess.run([g_train_op, error_g],feed_dict={real_boundary: batch_boundary})

      elapsed = time.time() - t

      assert not np.isnan(loss_value_g), 'Model diverged with loss = NaN'
      assert not np.isnan(loss_value_d), 'Model diverged with loss = NaN'

      if current_step%10 == 0:
        print("loss value g at " + str(loss_value_g))
        print("loss value d at " + str(loss_value_d))
        print("time per batch is " + str(elapsed))

      if current_step%100 == 0:
        summary_str = sess.run(summary_op, feed_dict={real_boundary: batch_boundary})
        summary_writer.add_summary(summary_str, current_step) 
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step)  
        print("saved to " + TRAIN_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
